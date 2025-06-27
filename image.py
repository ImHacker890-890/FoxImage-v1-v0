def generate():
    netG = Generator()
    netG.load_state_dict(torch.load("generator.pth"))
    netG.eval()

    with torch.no_grad():
        noise = torch.randn(64, 100, 1, 1)
        fake = netG(noise).detach().cpu()
        save_image(fake, "images.png", nrow=8, normalize=True)
