from cyclegan_pytorch.cyclegan import CycleGAN

model = CycleGAN('./data/a_dir', './data/hotdogs', epochs=10)

model.train(log_every=1)
model.translate_img('./data/a_dir/drone.jpg', 'res.jpg')
