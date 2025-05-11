from itertools import cycle
from numpy.random import randint
from pygame import Rect, init, time, display
from pygame.event import pump
from pygame.image import load
from pygame.surfarray import array3d, pixels_alpha
from pygame.transform import rotate
import numpy as np
import pygame
pygame.font.init()
font = pygame.font.SysFont("Arial", 32, bold=True)

class FlappyBird(object):
    init()
    dongHoFPS = time.Clock()
    chieuRongManHinh = 288
    chieuCaoManHinh = 512
    manHinh = display.set_mode((chieuRongManHinh, chieuCaoManHinh))
    display.set_caption('Deep Q-Network Flappy Bird')
    anhNen = load('assets/sprites/background-black.png').convert()
    anhNenDat = load('assets/sprites/base.png').convert_alpha()
    anhOng = [rotate(load('assets/sprites/pipe-green.png').convert_alpha(), 180),
              load('assets/sprites/pipe-green.png').convert_alpha()]
    anhChim = [load('assets/sprites/bird-upflap.png').convert_alpha(),
               load('assets/sprites/bird-midflap.png').convert_alpha(),
               load('assets/sprites/bird-downflap.png').convert_alpha()]
    matVaChamChim = [pixels_alpha(image).astype(bool) for image in anhChim]
    matVaChamOng = [pixels_alpha(image).astype(bool) for image in anhOng]
    tocDoKhungHinh = 30
    khoangCachOng = 100
    tocDoOng = -4
    tocDoRoiToiThieu = -8
    tocDoRoiToiDa = 10
    tocDoRoi = 1
    tocDoNhay = -9
    chiSoAnhChim = cycle([0, 1, 2, 1])

    def __init__(self):
        self.dem = self.chiSoChim = self.diemSo = 0
        self.gocXoay = 0
        self.rongChim = self.anhChim[0].get_width()
        self.caoChim = self.anhChim[0].get_height()
        self.rongOng = self.anhOng[0].get_width()
        self.caoOng = self.anhOng[0].get_height()
        self.viTriChimX = int(self.chieuRongManHinh / 5)
        self.viTriChimY = int((self.chieuCaoManHinh - self.caoChim) / 2)
        self.viTriDatX = 0
        self.viTriDatY = self.chieuCaoManHinh * 0.79
        self.doDichDat = self.anhNenDat.get_width() - self.anhNen.get_width()
        ong = [self.taoOng(), self.taoOng()]
        ong[0]["x_tren"] = ong[0]["x_duoi"] = self.chieuRongManHinh
        ong[1]["x_tren"] = ong[1]["x_duoi"] = self.chieuRongManHinh * 1.5
        self.danhSachOng = ong
        self.tocDoHienTai = 0
        self.daNhay = False

    def taoOng(self):
        x = self.chieuRongManHinh + 10
        gap_y = randint(2, 10) * 10 + int(self.viTriDatY / 5)
        return {
            "x_tren": x,
            "y_tren": gap_y - self.caoOng,
            "x_duoi": x,
            "y_duoi": gap_y + self.khoangCachOng
        }

    def kiemTraVaCham(self):
        if self.caoChim + self.viTriChimY + 1 >= self.viTriDatY:
            return True
        chim_bbox = Rect(self.viTriChimX, self.viTriChimY, self.rongChim, self.caoChim)
        hopOng = []
        for ong in self.danhSachOng:
            hopOng.append(Rect(ong["x_tren"], ong["y_tren"], self.rongOng, self.caoOng))
            hopOng.append(Rect(ong["x_duoi"], ong["y_duoi"], self.rongOng, self.caoOng))
            if chim_bbox.collidelist(hopOng) == -1:
                return False
            for i in range(2):
                vungGiao = chim_bbox.clip(hopOng[i])
                x1 = vungGiao.x - chim_bbox.x
                y1 = vungGiao.y - chim_bbox.y
                x2 = vungGiao.x - hopOng[i].x
                y2 = vungGiao.y - hopOng[i].y
                if np.any(self.matVaChamChim[self.chiSoChim][x1:x1 + vungGiao.width, y1:y1 + vungGiao.height] *
                          self.matVaChamOng[i][x2:x2 + vungGiao.width, y2:y2 + vungGiao.height]):
                    return True
        return False

    def capNhatKhungHinh(self, hanhDong):
        pump()
        thuong = 0.1
        ketThuc = False
        if hanhDong == 1:
            self.tocDoHienTai = self.tocDoNhay
            self.daNhay = True

        tamChimX = self.viTriChimX + self.rongChim / 2
        for ong in self.danhSachOng:
            tamOngX = ong["x_tren"] + self.rongOng / 2
            if tamOngX < tamChimX < tamOngX + 5:
                self.diemSo += 1
                thuong = 1
                break

        tocDoFlap = 3 if self.tocDoHienTai < 0 else 5
        if (self.dem + 1) % tocDoFlap == 0:
            self.chiSoChim = next(self.chiSoAnhChim)
            self.dem = 0
        self.viTriDatX = -((-self.viTriDatX + 100) % self.doDichDat)

        if self.tocDoHienTai < self.tocDoRoiToiDa and not self.daNhay:
            self.tocDoHienTai += self.tocDoRoi
        if self.daNhay:
            self.daNhay = False

        self.viTriChimY += min(self.tocDoHienTai, self.viTriChimY - self.tocDoHienTai - self.caoChim)
        if self.viTriChimY < 0:
            self.viTriChimY = 0

        for ong in self.danhSachOng:
            ong["x_tren"] += self.tocDoOng
            ong["x_duoi"] += self.tocDoOng
        if 0 < self.danhSachOng[0]["x_duoi"] < 5:
            self.danhSachOng.append(self.taoOng())
        if self.danhSachOng[0]["x_duoi"] < -self.rongOng:
            del self.danhSachOng[0]
        if self.kiemTraVaCham():
            ketThuc = True
            thuong = -1
            self.__init__()

        self.manHinh.blit(self.anhNen, (0, 0))
        self.manHinh.blit(self.anhNenDat, (self.viTriDatX, self.viTriDatY))

        gocXoayMucTieu = max(-90, min(25, -self.tocDoHienTai * 3))
        self.gocXoay += (gocXoayMucTieu - self.gocXoay) * 0.2
        anhChim = self.anhChim[self.chiSoChim]
        anhXoay = rotate(anhChim, self.gocXoay)
        vungAnh = anhXoay.get_rect(center=(self.viTriChimX + anhChim.get_width() // 2,
                                           self.viTriChimY + anhChim.get_height() // 2))
        self.manHinh.blit(anhXoay, vungAnh.topleft)

        for ong in self.danhSachOng:
            self.manHinh.blit(self.anhOng[0], (ong["x_tren"], ong["y_tren"]))
            self.manHinh.blit(self.anhOng[1], (ong["x_duoi"], ong["y_duoi"]))

        diem_surface = font.render(str(self.diemSo), True, (255, 255, 255))
        diem_rect = diem_surface.get_rect(center=(self.chieuRongManHinh // 2, 50))
        self.manHinh.blit(diem_surface, diem_rect)

        anh = array3d(display.get_surface())
        display.update()
        self.dongHoFPS.tick(self.tocDoKhungHinh)
        return anh, thuong, ketThuc
