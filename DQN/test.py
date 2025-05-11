import argparse
import os
import torch

from src.deep_q_network import DeepQNetwork
from src.flappy_bird import FlappyBird
from src.utils import xuLyAnh

def docThamSo():
    parser = argparse.ArgumentParser("Kiểm tra mô hình Deep Q Network đã huấn luyện với Flappy Bird")
    parser.add_argument("--kich_thuoc_anh", type=int, default=84)
    parser.add_argument("--duong_luu", type=str, default="trained_models")
    parser.add_argument("--ten_model", type=str, default="flappy_bird_2000000_state_dict.pt", help="Tên file model .pt đã lưu bằng state_dict()")
    return parser.parse_args()

def choiMotVan(game, model, device, kich_thuoc_anh):
    anh, _, _ = game.capNhatKhungHinh(0)
    anh = xuLyAnh(anh[:game.chieuRongManHinh, :int(game.viTriDatY)], kich_thuoc_anh, kich_thuoc_anh)
    anh = torch.from_numpy(anh).to(device)
    trangThai = torch.cat([anh for _ in range(4)])[None, :, :, :]

    tong_diem = 0
    so_ong = 0

    while True:
        with torch.no_grad():
            duDoan = model.tienTrinh(trangThai)[0]
        hanhDong = torch.argmax(duDoan).item()

        anhTiep, thuong, ketThuc = game.capNhatKhungHinh(hanhDong)
        if thuong == 1:
            so_ong += 1
        tong_diem += thuong

        anhTiep = xuLyAnh(anhTiep[:game.chieuRongManHinh, :int(game.viTriDatY)], kich_thuoc_anh, kich_thuoc_anh)
        anhTiep = torch.from_numpy(anhTiep).to(device)
        trangThai = torch.cat((trangThai[0, 1:, :, :], anhTiep)).unsqueeze(0)

        if ketThuc:
            print(f" Game over – Tổng reward: {tong_diem:.2f} | Số ống vượt qua: {so_ong}")
            break

def kiemThu(tuyChon):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(123)

    duong_dan_model = os.path.join(tuyChon.duong_luu, tuyChon.ten_model)
    if not os.path.isfile(duong_dan_model):
        print(f"Không tìm thấy file mô hình: {duong_dan_model}")
        return

    print(f"Đang tải model từ: {duong_dan_model}")
    model = DeepQNetwork().to(device)
    model.load_state_dict(torch.load(duong_dan_model, map_location=device))
    model.eval()

    game = FlappyBird()

    so_van = 0
    while True:
        so_van += 1
        print(f"\n BẮT ĐẦU VÁN {so_van}")
        choiMotVan(game, model, device, tuyChon.kich_thuoc_anh)

if __name__ == "__main__":
    tuyChon = docThamSo()
    kiemThu(tuyChon)
