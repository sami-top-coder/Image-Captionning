import torch
import torchvision.transforms as transforms
from PIL import Image


def print_examples(model, device, dataset):
    transform = transforms.Compose(
        [
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    model.eval()
    test_img1 = transform(Image.open(r"C:\Users\Asus\Desktop\Custom data\Test Data\10815824_2997e03d76.jpg").convert("RGB")).unsqueeze(
        0
    )
    print("Example 1 CORRECT: Dog on a beach by the ocean")
    # print(
    #     "Example 1 OUTPUT: "
    #     + " ".join(model.caption_image(test_img1.to(device), dataset.vocab))
    # )
    test_img2 = transform(
        Image.open(r"C:\Users\Asus\Desktop\Custom data\Test Data\10815824_2997e03d76.jpg").convert("RGB")
    ).unsqueeze(0)
    print("Example 2 CORRECT: Child holding red frisbee outdoors")
    # print(
    #     "Example 2 OUTPUT: "
    #     + " ".join(model.caption_image(test_img2.to(device), dataset.vocab))
    # )
    test_img3 = transform(Image.open(r"C:\Users\Asus\Desktop\Custom data\Test Data\667626_18933d713e.jpg").convert("RGB")).unsqueeze(
        0
    )
    # print(r"C:\Users\Asus\Desktop\Custom data\Test Data\667626_18933d713e.jpg")
    # print(
    #     "Example 3 OUTPUT: "
    #     + " ".join(model.caption_image(test_img3.to(device), dataset.vocab))
    # )
    test_img4 = transform(
        Image.open(r"C:\Users\Asus\Desktop\Custom data\Test Data\667626_18933d713e.jpg").convert("RGB")
    ).unsqueeze(0)
    print("Example 4 CORRECT: A small boat in the ocean")
    # print(
    #     "Example 4 OUTPUT: "
    #     + " ".join(model.caption_image(test_img4.to(device), dataset.vocab))
    # )
    test_img5 = transform(
        Image.open(r"C:\Users\Asus\Desktop\Custom data\Test Data\667626_18933d713e.jpg").convert("RGB")
    ).unsqueeze(0)
    print("Example 5 CORRECT: A cowboy riding a horse in the desert")
    # print(
    #     "Example 5 OUTPUT: "
    #     + " ".join(model.caption_image(test_img5.to(device), dataset.vocab))
    # )
    model.train()


def save_checkpoint(state, filename="my_checkpoint.pth.tar"): # save checkpoint
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer): # fonction load_checkpoint 3adia ta3 3ada
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    step = checkpoint["step"]
    return step