using Base.Iterators: partition
using Flux
using Flux.Optimise: update!
using Flux: logitbinarycrossentropy
using Images
using Statistics
using Parameters: @with_kw
using Printf
using Random
using CUDA
using Pkg
using BSON
using Plots

# 导入data_loader包用于加载数据
include("data_loader.jl")

# GPU使用
gpu_id = 1  ## set < 0 for no cuda, >= 0 for using a specific device (if available)
if CUDA.has_cuda_gpu() && gpu_id >=0
    device!(gpu_id)
    device = Flux.gpu
    @info "Training on GPU-$(gpu_id)"
else
    device = Flux.cpu
    @info "Training on CPU"
end

# 参数定义
using Parameters: @with_kw
@with_kw mutable struct Args
    batch_size::Int = 128
    latent_dim::Int = 100   # DCGAN中由latent_dim维noise生成一张图像
    epochs::Int = 300
    verbose_freq::Int = 5000  #打印训练信息的频率，即每隔多少次训练（batch而非epoch）打印一次训练信息。
    output_x::Int = 3       # 生成output_x行output_y列图像，在这里我们生成1张
    output_y::Int = 3
    lr_dscr::Float64 = 0.0002  # 学习率遵照DCGAN论文设置为0.0002，论文中提到这样可以使训练稳定
    lr_gen::Float64 = 0.0002
end

# 图像生成及显示处理代码
function create_output_image(gen, fixed_noise, args)    
    # @eval Flux.istraining() = false
    testmode!(gen)
    fake_images = @. cpu(gen(fixed_noise))
    testmode!(gen, false)
    # @eval Flux.istraining() = true
    println(size(fake_images))
    println(size(fake_images[1]))
    # 定义大图像
    grid_size = (args.output_x, args.output_y)
    big_image = Array{RGB{Float32}, 2}(undef, 64*args.output_y, 64*args.output_x)
    # 将生成的小图像填充到大图像中
    for i in 1:args.output_x
        for j in 1:args.output_y
            idx = (i-1)*args.output_y + j
            println(idx)
            small_image = colorview(RGB, reshape(fake_images[idx], 3, 64, 64))
            display(colorview(RGB, small_image))
            big_image[(1:64) .+ (i-1)*64, (1:64) .+ (j-1)*64] .= small_image
        end
    end
    return big_image
end

# 借鉴DCGAN原始论文提到的参数初始化方法进行参数初始化
# weight initialization as given in the paper https://arxiv.org/abs/1511.06434
dcgan_init(shape...) = randn(Float32, shape...) * 0.02f0

# 判别器模型定义，整体与DCGAN论文中相同，使用了leakyrelu
function Discriminator()
    return Chain(
        Conv((4, 4), 3=>128, stride=2, pad=1, init = dcgan_init), # 输入 64*64*3
        x -> leakyrelu.(x, 0.2),
        Conv((4, 4), 128=>256, stride=2, pad=1, init = dcgan_init),    # 输入 32*32*128
        BatchNorm(256),
        x -> leakyrelu.(x, 0.2),
        Conv((4, 4), 256=>512, stride=2, pad=1, init = dcgan_init), # 输入 16*16*256
        BatchNorm(512),
        x -> leakyrelu.(x, 0.2),
        Conv((4, 4), 512=>1024, stride=2, pad=1, init = dcgan_init), # 输入 8*8*512
        BatchNorm(1024),
        x -> leakyrelu.(x, 0.2),
        Flux.flatten,
        Dense(4 * 4 * 1024, 1),
        # sigmoid     # 经实验，使用sigmoid会导致模型无法学习到有用的信息，所以不进行使用
    )
end

# 生成器模型定义，使用ConvTranspose反卷积生成图像
function Generator(latent_dim)
    return Chain(
        Dense(latent_dim, 4*4*1024),
        BatchNorm(4*4*1024, relu),
        x -> reshape(x, 4, 4, 1024, :),
        ConvTranspose((4, 4), 1024=>512, stride=2, pad=1, init = dcgan_init),
        BatchNorm(512, relu),
        ConvTranspose((4, 4), 512=>256, stride=2, pad=1, init = dcgan_init),
        BatchNorm(256, relu),
        ConvTranspose((4, 4), 256=>128, stride=2, pad=1, init = dcgan_init),
        BatchNorm(128, relu),
        ConvTranspose((4, 4), 128=>3, stride=2, pad=1, init = dcgan_init),  # 修改输出通道为3，RGB
        x -> tanh.(x)
    )
end

# 定义判别器loss，使用logitbinarycrossentropy，让判别器学会分别真实图像与生成器生成的图像
function discriminator_loss(real_output, fake_output)
    real_loss = mean(logitbinarycrossentropy.(real_output, 1f0))
    fake_loss = mean(logitbinarycrossentropy.(fake_output, 0f0))
    return (real_loss + fake_loss)
end

# 生成器loss，同样使用logitbinarycrossentropy，目的是生成一张图像尽可能让判别器分辨不出来是否是审生成的
generator_loss(fake_output) = mean(logitbinarycrossentropy.(fake_output, 1f0))

# 训练判别器
function train_discriminator!(gen, dscr, x, opt_dscr, args)
    Flux.trainmode!(dscr, true)  # 使模型处于可以训练的状态
    noise = randn!(similar(x, (args.latent_dim, args.batch_size))) 
    fake_input = gen(noise)
    ps = Flux.params(dscr)   # 获取模型参数
    loss, back = Flux.pullback(ps) do
        discriminator_loss(dscr(x), dscr(fake_input))
    end
    gradients = Flux.gradient(() -> discriminator_loss(dscr(x), dscr(fake_input)), Flux.params(dscr)) # 获取梯度
    Flux.Optimise.update!(opt_dscr, Flux.params(dscr), gradients)
    return loss
end

# 训练生成器
function train_generator!(gen, dscr, x, opt_gen, args)
    Flux.trainmode!(gen, true) # 使模型处于可以训练的状态
    noise = randn!(similar(x, (args.latent_dim, args.batch_size)))  #生成一个noise
    ps = Flux.params(gen) # 加载模型参数
    loss, back = Flux.pullback(ps) do
        generator_loss(dscr(gen(noise)))
    end
    gradients = Flux.gradient(() -> generator_loss(dscr(gen(noise))), Flux.params(gen))  # 获取梯度
    Flux.Optimise.update!(opt_gen, Flux.params(gen), gradients)
    return loss
end

# 保存模型参数，这个经过实验发现虽然可以保存模型参数，但是使用保存的模型参数时并不能使用，不知为何
function save_model_params(model, filename)
    params_dict = Flux.params(model)
    BSON.@save filename params_dict
end

# 暂存loss函数用于显示
gen_loss = []
dscr_loss = []

function train(; kws...)
    # Model Parameters
    args = Args(; kws...)
    # 不能使用MLDatasets，所以加载自己的数据集
    train_data = load_dataset_as_batches("/home/wyj/code/julia-ai/GAN/Data/img_align_celeba/")
    images = train_data
    println(size(images))
    # Normalize Data   # 不需要对图像进行标准化，处理数据集时使用了channelview，得到的image数组是在0-1之间的
    # println(images)
    # images /= 255.0
    image_tensor = reshape(images[:, :, :, :], 64, 64, 3, :)
    # Partition into batches
    data = [image_tensor[:, :, :, r] |> device for r in partition(1:30000, args.batch_size)]
    fixed_noise = [randn(args.latent_dim, 1) |> device for _=1:args.output_x*args.output_y]
    # Discriminator
    d_model = Discriminator() |> device
    # Generator
    g_model = Generator(args.latent_dim) |> device
    # Optimizers
    opt_dscr = ADAM(args.lr_dscr, (0.5, 0.999))  #这里(0.5, 0.999)中原本应为0.9，DCGAN中提到将动量项β1设为0.5有助于GAN训练的稳定
    opt_gen = ADAM(args.lr_gen, (0.5, 0.999))

    # 记录 step的loss
    loss_Gen_value = 0.0
    loss_Dscr_value = 0.0
    # Training
    train_steps = 0
    for ep in 1:args.epochs
         # 记录 eopch的loss
        coef = 0
        loss_Gen_Epoch = 0.0
        loss_Dscr_Epoch = 0.0
        # 记录每轮开始时间
        epoch_start_time = time()
        @info "Epoch $ep"
        for x in data
            loss_Gen_value = train_discriminator!(g_model, d_model, x, opt_dscr, args)
            loss_Dscr_value = train_generator!(g_model, d_model, x, opt_gen, args)
            loss_Gen_Epoch += loss_Gen_value
            loss_Dscr_Epoch += loss_Dscr_value

            if train_steps % args.verbose_freq == 0
                @info("Train step $(train_steps), Discriminator loss = $(loss_Dscr_value), Generator loss = $(loss_Gen_value)")
                # Show generated fake image
                output_image = create_output_image(g_model, fixed_noise, args)
                display(colorview(RGB, output_image))
            end
            coef += 1
            train_steps += 1
        end
        # 将每一轮的loss存在数组中
        push!(gen_loss, loss_Gen_Epoch / coef)
        push!(dscr_loss, loss_Dscr_Epoch / coef)
        # 记录每轮结束时间
        epoch_end_time = time()
        epoch_time = epoch_end_time - epoch_start_time
        println(" - Epoch Time: $epoch_time seconds")
    end

    # save_model_params(g_model, "model_weights.bson")  # 保存模型参数，但是似乎没有用
    output_image = create_output_image(g_model, fixed_noise, args)
    display(colorview(RGB, output_image))

    # 绘制生成器loss
    plot1 = plot(1:length(gen_loss), gen_loss, xlabel="Epoch", ylabel="Gen_Loss", label="gen_loss", legend=:topright)
    # 显示第一张图形
    display(plot1)

    # 绘制判别器loss
    plot2 = plot(1:length(dscr_loss), dscr_loss, xlabel="Epoch", ylabel="Dscr_Loss", label="dscr_loss", legend=:topright)
    # 显示第二张图形
    display(plot2)
end
train()
