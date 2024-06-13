import os
import torch
import wandb
from torch.nn import MSELoss
from datetime import date
from glob import glob

def get_device():
    """
    Возвращает устройство (GPU, если доступен, в противном случае CPU).

    Returns:
        torch.device: Устройство (cuda или cpu).
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def instantiate_model(model, dataset, encoding_dim, **kwargs):
    """
    Создает экземпляр заданной модели с указанными параметрами.

    Args:
        model (torch.nn.Module): Нейронная сеть модели.
        dataset (torch.Tensor): Обучающий набор данных.
        encoding_dim (int): Размерность кодирования.
        **kwargs: Дополнительные именованные аргументы для передачи модели.

    Returns:
        torch.nn.Module: Экземпляр созданной модели.
    """
    return model(dataset[-1].shape[-1], encoding_dim, **kwargs)


def train_model(model, train_set, eval_set, verbose, lr, epochs, clip_value, device=None, wandb_project="", save_path="../models"):
    """
    Обучает заданную модель на предоставленном обучающем наборе данных.

    Args:
        model (torch.nn.Module): Нейронная сеть модели.
        train_set (Iterable[torch.Tensor]): Обучающий набор данных.
        eval_set (Iterable[torch.Tensor]): Валидационный набор данных.
        verbose (bool): Если True, выводит прогресс обучения.
        lr (float): Скорость обучения для оптимизации.
        epochs (int): Количество эпох обучения.
        clip_value (float or None): Значение для обрезки градиентов, или None, если не применяется.
        device (torch.device or None): Устройство для обучения.

    Returns:
        list: Средние значения потерь для каждой эпохи.
    """
    if device is None:
        device = get_device()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = MSELoss(reduction="sum")
    if len(wandb_project) > 0:
        wandb.init(
            project=wandb_project,
            # track hyperparameters
            config={
            "learning_rate": lr,
            "architecture": model.__class__.__name__,
            "dataset shape": str(train_set[0].shape),
            "epochs": epochs,
            }
        )

    mean_losses_train = []
    mean_losses_val = []
    best_val_loss = float('inf')
    for epoch in range(1, epochs + 1):
        
        #train
        model.train()
        losses_tr = []
        for x in train_set:
            x = x.to(device)
            optimizer.zero_grad()
            x_pred = model(x)
            loss = criterion(x_pred, x)
            loss.backward()
            if clip_value is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
            optimizer.step()
            losses_tr.append(loss.item())

        #eval
        model.eval()
        losses_val = []
        for y in eval_set:
            with torch.no_grad():
                y = y.to(device)
                y_pred = model(y)
                loss = criterion(x_pred, x)
                losses_val.append(loss.item())
                
        mean_loss_tr = torch.mean(torch.tensor(losses_tr))
        mean_loss_val = torch.mean(torch.tensor(losses_val))
        mean_losses_train.append(mean_loss_tr.item())
        mean_losses_val.append(mean_loss_val.item())
        
        # сохраняем лучшую модель
        if best_val_loss > mean_loss_val:
            best_val_loss = mean_loss_val
            # удаляем предыдущее сохранение
            for file in glob(f'{save_path}/{model.__class__.__name__}-{date.today()}*'):
                os.remove(file)
            # сохраняем модель
            torch.save(
                model.state_dict(), 
                f'{save_path}/{model.__class__.__name__}-{date.today()}-val_loss:{round(mean_loss_val.item(), 4)}.pth'
            )
        if verbose:
            print(f"Epoch: {epoch}, Train loss: {mean_loss_tr}, Valid loss: {mean_loss_val}")
        if len(wandb_project) > 0:
            wandb.log({"Train loss": mean_loss_tr, "Valid loss": mean_loss_val})
    
    if len(wandb_project) > 0:
        wandb.finish()
    return mean_losses_train, mean_losses_val
    
def get_preds(model, dataset, device=None):
    """
    Получает ответы модели, сгенерированные моделью для предоставленного обучающего набора данных.

    Args:
        model (torch.nn.Module): Нейронная сеть модели.
        dataset (Iterable[torch.Tensor]): Обучающий набор данных.
        device (torch.device or None): Устройство для использования при выводе.

    Returns:
        list: Список тензоров кодировок, соответствующих обучающему набору данных.
    """
    if device is None:
        device = get_device()
    model.eval()
    preds = [model(x.to(device)) for x in dataset]
    return preds


def train(
    model,
    train_set,
    eval_set,
    encoding_dim,
    verbose=False,
    lr=1e-3,
    epochs=50,
    clip_value=1,
    device=None,
    **kwargs,
):
    """
    Обучает модель автокодировщика на предоставленном обучающем наборе данных и возвращает соответствующую информацию.

    Args:
        model (torch.nn.Module): Модель автокодировщика.
        train_set (Iterable[torch.Tensor]): Обучающий набор данных.
        eval_set (Iterable[torch.Tensor]): Валидационный набор данных.
        encoding_dim (int): Размерность кодирования.
        verbose (bool): Если True, выводит прогресс обучения.
        lr (float): Скорость обучения для оптимизации.
        epochs (int): Количество эпох обучения.
        clip_value (float or None): Значение для обрезки градиентов, или None, если не применяется.
        device (torch.device or None): Устройство для обучения.
        **kwargs: Дополнительные именованные аргументы для создания модели.

    Returns:
        tuple: Кортеж, содержащий кодировщик, декодер, кодировки и потери.
    """
    model = instantiate_model(model, train_set, encoding_dim, **kwargs)
    losses_train, losses_val = train_model(
        model, train_set, eval_set, verbose, lr, epochs, clip_value, device
    )

    return model, losses_train, losses_val