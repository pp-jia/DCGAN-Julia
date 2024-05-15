using Flux
using BSON
using Images
using CUDA
using Random

# 参数定义
using Parameters: @with_kw
@with_kw mutable struct Args
    batch_size::Int = 128
    latent_dim::Int = 100
    epochs::Int = 100
    verbose_freq::Int = 5000
    # 想要通过我们预训练好的模型生成新的图像，在这里可以指定想要生成几张4*4,表示生成16张
    output_x::Int = 3       
    output_y::Int = 3
    lr_dscr::Float64 = 0.0002  
    lr_gen::Float64 = 0.0002
end
# Model Parameters
args = Args()

function Generator(latent_dim)
    return Chain(
        Dense(latent_dim, 4*4*1024),
        BatchNorm(4*4*1024, relu),
        x -> reshape(x, 4, 4, 1024, :),
        ConvTranspose((4, 4), 1024=>512, stride=2, pad=1),
        BatchNorm(512, relu),
        ConvTranspose((4, 4), 512=>256, stride=2, pad=1),
        BatchNorm(256, relu),
        ConvTranspose((4, 4), 256=>128, stride=2, pad=1),
        BatchNorm(128, relu),
        ConvTranspose((4, 4), 128=>3, stride=2, pad=1),  # 修改输出通道为3，RGB
        x -> tanh.(x)
    )
end

# 显示生成图像
function create_output_image(gen, fixed_noise, args)    
    # @eval Flux.istraining() = false
    # testmode!(gen, true)
    fake_images = @. cpu(gen(fixed_noise))
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

# 加载模型参数
function load_model_params(model, filename)
    BSON.@load filename params_dict
    Flux.loadparams!(model, params_dict)
end


fixed_noise = [randn(args.latent_dim, 1) |> Flux.cpu for _=1:args.output_x*args.output_y]

# Generator
g_model = Generator(100)

load_model_params(g_model, "/home/wyj/code/julia-ai/model_weights300.bson")

output_image = create_output_image(g_model, fixed_noise, args)
display(colorview(RGB, output_image))