using Base.Iterators: partition
using Images

# 加载数据集
function img_load(img_name)
    img = load(img_name)
    img = imresize(img, 64, 64)
    img = reshape(Float32.(channelview(img)), 64, 64, 3)
    if(img_name == "/home/wyj/code/julia-ai/GAN/img_align_celeba/004512.jpg")  # 显示数据集中的某一个图像
        img_display = img
        println(img_display)
        img_display = reshape(img_display, 3, 64, 64)
        display(colorview(RGB, img_display))
    end
    return img
end

function load_dataset_as_batches(path)
    imges = Array{Float32,4}(undef, 64, 64, 3, 30000)
    data = []
    i = 1
    for r in readdir(path)
        if i == 30000
            break
        end
        img_path = string(path, r)
        imges[:, :, :, i] = img_load(img_path)
        i += 1
    end
    return imges
end