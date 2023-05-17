# import random
# import pygame
# import numpy as np
#
#
# pygame.init()
#
# sc_width = 600
# sc_height = 800
# circle_r = 40
# rectangle_width  =  200
# rectangle_height  =  80
# colors = [(100,90,160),(0,0,0)]
# fps = pygame.time.Clock()
# screen  =pygame.display.set_mode((sc_width,sc_height))
# birim = 0
# sonuc = []
# data = []
# bitis = 0
#
# class rectangle:
#     def __init__(self):
#         self.width = rectangle_width
#         self.height = rectangle_height
#         self.x  = sc_width // 2
#         self.y = sc_height - self.height
#
#     def draw_rectangle(self):
#         pygame.draw.rect(screen,colors[1],[self.x,self.y,self.width,self.height])
#
#
#
# class circle:
#     def __init__(self):
#         self.r = circle_r
#         self.x = random.randint(self.r,sc_width-self.r)
#         self.y = 0
#     def draw_circle(self):
#         pygame.draw.circle(screen,colors[1],[self.x,self.y],self.r)
#
#
#
# new_rectangle = rectangle()
# new_circle = circle()
#
#
#
# while True:
#
#     new_circle.y += 20
#
#     if(new_circle.y >= sc_height):
#         new_circle.y = 0
#         new_circle.x = random.randint(circle_r,sc_width - circle_r)
#
#     if bitis == 9:
#         pygame.quit()
#         break
#
#
#
#     for event in pygame.event.get():
#         if event.type == pygame.KEYDOWN:
#             if(event.key == pygame.K_LEFT):
#                 birim = -15
#             if (event.key == pygame.K_RIGHT):
#                 birim = 15
#             if (event.key == pygame.K_q):
#                 pygame.quit()
#         if event.type == pygame.KEYUP:
#             if(event.key == pygame.K_LEFT or event.key == pygame.K_RIGHT):
#                 birim = 0
#
#
#     if new_rectangle.x + birim < 0 or new_rectangle.x + birim + rectangle_width > sc_width:
#         pass
#     else:
#         new_rectangle.x += birim
#
#     if new_circle.y > sc_height - rectangle_height:
#         if new_circle.x >= new_rectangle.x and new_circle.x <= new_rectangle.x + rectangle_width:
#             bitis += 1
#
#
#     screen.fill(colors[0])
#
#     if birim != 0:
#         data += [[new_rectangle.x,new_circle.x,new_circle.y]]
#         sonuc += [birim]
#
#     new_rectangle.draw_rectangle()
#     new_circle.draw_circle()
#
#     pygame.display.update()
#
#     fps.tick(60)
#
# kayit_giris = "kayit_giris"
# kayit_cikis = "kayit_cikis"
#
# try:
#     np.save(kayit_giris,np.concatenate((np.load(kayit_giris),np.array(data).reshape(-1,3))))
#     np.save(kayit_cikis, np.concatenate((np.load(kayit_cikis), np.array(sonuc).reshape(-1, ))))
# except:
#     np.save(kayit_giris, np.array(data).reshape(-1, 3))
#     np.save(kayit_cikis, np.array(sonuc).reshape(-1, ))
#
#
#
#
