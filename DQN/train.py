import argparse
import os
import shutil
from random import random, randint, sample

import numpy as np
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter

from src.deep_q_network import DeepQNetwork
from src.flappy_bird import FlappyBird
from src.utils import xuLyAnh
import torch.serialization

torch.serialization.add_safe_globals([DeepQNetwork])  # Cho phép load object từ file .pt

def docThamSo():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kich_thuoc_anh", type=int, default=84)
    parser.add_argument("--kich_thuoc_lo", type=int, default=32)
    parser.add_argument("--toi_uu", type=str, choices=["sgd", "adam"], default="adam")
    parser.add_argument("--toc_do_hoc", type=float, default=1e-6)
    parser.add_argument("--he_so_giam", type=float, default=0.99)
    parser.add_argument("--epsilon_dau", type=float, default=0.1)
    parser.add_argument("--epsilon_cuoi", type=float, default=1e-4)
    parser.add_argument("--so_vong_lap", type=int, default=2000000)
    parser.add_argument("--kich_thuoc_bo_nho", type=int, default=50000)
    parser.add_argument("--duong_log", type=str, default="tensorboard")
    parser.add_argument("--duong_luu", type=str, default="trained_models")
    return parser.parse_args()

def taiMoHinhTuCheckpoint(duong_dan):
    if not os.path.exists(duong_dan):
        return None, 0

    cac_file = []
    for f in os.listdir(duong_dan):
        if f.startswith("flappy_bird_") and f.endswith(".pt"):
            ten = f.replace("flappy_bird_", "").replace(".pt", "")
            if ten.isdigit():
                cac_file.append((int(ten), f))

    if not cac_file:
        return None, 0

    cac_file.sort()
    vong_bat_dau, tep_moi_nhat = cac_file[-1]

    mo_hinh = torch.load(os.path.join(duong_dan, tep_moi_nhat), map_location="cuda" if torch.cuda.is_available() else "cpu", weights_only=False)
    return mo_hinh, vong_bat_dau

def huanLuyen(tuyChon):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)

    moHinh, vong = taiMoHinhTuCheckpoint(tuyChon.duong_luu)
    if moHinh is None:
        moHinh = DeepQNetwork()
        vong = 0

    if torch.cuda.is_available():
        moHinh.cuda()

    if os.path.isdir(tuyChon.duong_log):
        shutil.rmtree(tuyChon.duong_log)
    os.makedirs(tuyChon.duong_log)

    writer = SummaryWriter(tuyChon.duong_log)
    toiUu = torch.optim.Adam(moHinh.parameters(), lr=tuyChon.toc_do_hoc)
    matMat = nn.MSELoss()
    game = FlappyBird()
    anh, thuong, ketThuc = game.capNhatKhungHinh(0)
    anh = xuLyAnh(anh[:game.chieuRongManHinh, :int(game.viTriDatY)], tuyChon.kich_thuoc_anh, tuyChon.kich_thuoc_anh)
    anh = torch.from_numpy(anh)
    if torch.cuda.is_available():
        anh = anh.cuda()
    trangThai = torch.cat(tuple(anh for _ in range(4)))[None, :, :, :]

    boNho = []
    while vong < tuyChon.so_vong_lap:
        duDoan = moHinh.tienTrinh(trangThai)[0]
        epsilon = tuyChon.epsilon_cuoi + (
            (tuyChon.so_vong_lap - vong) * (tuyChon.epsilon_dau - tuyChon.epsilon_cuoi) / tuyChon.so_vong_lap)
        u = random()
        hanhDong = randint(0, 1) if u <= epsilon else torch.argmax(duDoan).item()

        anhTiep, thuong, ketThuc = game.capNhatKhungHinh(hanhDong)
        anhTiep = xuLyAnh(anhTiep[:game.chieuRongManHinh, :int(game.viTriDatY)], tuyChon.kich_thuoc_anh, tuyChon.kich_thuoc_anh)
        anhTiep = torch.from_numpy(anhTiep)
        if torch.cuda.is_available():
            anhTiep = anhTiep.cuda()
        trangThaiTiep = torch.cat((trangThai[0, 1:, :, :], anhTiep))[None, :, :, :]

        boNho.append([trangThai, hanhDong, thuong, trangThaiTiep, ketThuc])
        if len(boNho) > tuyChon.kich_thuoc_bo_nho:
            del boNho[0]

        loMau = sample(boNho, min(len(boNho), tuyChon.kich_thuoc_lo))
        ts, hd, t, tst, kt = zip(*loMau)
        ts = torch.cat(tuple(s for s in ts))
        hd = torch.from_numpy(np.array([[1, 0] if h == 0 else [0, 1] for h in hd], dtype=np.float32))
        t = torch.from_numpy(np.array(t, dtype=np.float32)[:, None])
        tst = torch.cat(tuple(s for s in tst))

        if torch.cuda.is_available():
            ts, hd, t, tst = ts.cuda(), hd.cuda(), t.cuda(), tst.cuda()

        duDoanHienTai = moHinh.tienTrinh(ts)
        duDoanTiep = moHinh.tienTrinh(tst)

        y = torch.cat(tuple(thuong if done else thuong + tuyChon.he_so_giam * torch.max(pred)
                            for thuong, done, pred in zip(t, kt, duDoanTiep)))

        q_value = torch.sum(duDoanHienTai * hd, dim=1)
        toiUu.zero_grad()
        loss = matMat(q_value, y)
        loss.backward()
        toiUu.step()

        trangThai = trangThaiTiep
        vong += 1
        print("Vong: {}/{}, HanhDong: {}, MatMat: {}, Epsilon: {}, Thuong: {}, Q-max: {}".format(
            vong + 1, tuyChon.so_vong_lap, hanhDong, loss.item(), epsilon, thuong, torch.max(duDoan).item()))

        writer.add_scalar('Train/Loss', loss.item(), vong)
        writer.add_scalar('Train/Epsilon', epsilon, vong)
        writer.add_scalar('Train/Reward', thuong, vong)
        writer.add_scalar('Train/Q-value', torch.max(duDoan).item(), vong)

        if (vong + 1) % 500000 == 0:
            torch.save(moHinh, f"{tuyChon.duong_luu}/flappy_bird_{vong+1}.pt")

        torch.save(moHinh, f"{tuyChon.duong_luu}/flappy_bird.pt")

if __name__ == "__main__":
    tuyChon = docThamSo()
    huanLuyen(tuyChon)
