import jittor as jt
from jittor.dataset import DataLoader
from models.mask_rcnn import MaskRCNN
import configparser
from pathlib import Path

config = configparser.ConfigParser()
config.read("config.cfg")


def train(model, train_loader, optimizer, num_epochs):
    """
    训练Mask R-CNN模型

    Args:
        model: Mask R-CNN模型实例
        train_loader: 训练数据加载器
        optimizer: 优化器实例
        num_epochs: 训练轮数
    """
    model.train()

    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, (images, targets) in enumerate(train_loader):
            # 确保targets的格式正确
            targets = [{k: v for k, v in t.items()} for t in targets]

            # 前向传播和损失计算
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            total_loss += losses.item()

            # 反向传播和优化
            optimizer.zero_grad()
            optimizer.backward(losses)
            optimizer.step()

            # 打印训练信息
            if batch_idx % 10 == 0:
                print(
                    f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{len(train_loader)}], "
                    f"Loss: {losses.item():.4f}"
                )

        # 打印每个epoch的平均损失
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")

    # 保存模型
    save_path = Path(config["MODEL"]["save_path"])
    save_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(save_path))
    print(f"Model saved to {save_path}")


def main():
    # 设置设备
    jt.flags.use_cuda = config.getboolean("DEVICE", "use_cuda", fallback=True)

    # 初始化模型
    model = MaskRCNN(num_classes=config.getint("MODEL", "num_classes"))

    # 设置优化器
    optimizer = jt.optim.Adam(
        model.parameters(),
        lr=config.getfloat("TRAIN", "learning_rate", fallback=0.001),
        betas=(
            config.getfloat("TRAIN", "beta1", fallback=0.9),
            config.getfloat("TRAIN", "beta2", fallback=0.999),
        ),
        eps=config.getfloat("TRAIN", "eps", fallback=1e-8),
        weight_decay=config.getfloat("TRAIN", "weight_decay", fallback=0.0001),
    )

    # 创建数据加载器
    train_loader = DataLoader(
        dataset=config["DATASET"]["train_path"],
        batch_size=config.getint("TRAIN", "batch_size"),
        shuffle=True,
        num_workers=config.getint("TRAIN", "num_workers", fallback=4),
        drop_last=True,
    )

    # 训练模型
    num_epochs = config.getint("TRAIN", "num_epochs")
    train(model, train_loader, optimizer, num_epochs)


if __name__ == "__main__":
    main()
