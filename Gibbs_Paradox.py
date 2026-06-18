import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from math import factorial

print("=== Симуляция Парадокса Гиббса ===")
N_half = int(input("Введите число частиц в одной половине (рекомендуется 20-50): "))
is_same_gas = input("Газы одинаковые? (y/n): ").strip().lower() == 'y'
use_gibbs_correction = input("Добавить поправку на неразличимость (1/N!)? (y/n): ").strip().lower() == 'y'

N_total = N_half * 2
L_WIDTH, L_HEIGHT = 10.0, 10.0 #Размеры сосуда
WALL_FRAME = 50 #На каком кадре убирается перегородка
PARTICLE_RADIUS = 0.25 #Радиус частицы


x_left = np.random.uniform(0.5, 4.5, N_half)
y_left = np.random.uniform(0.5, 9.5, N_half)
x_right = np.random.uniform(5.5, 9.5, N_half)
y_right = np.random.uniform(0.5, 9.5, N_half)

x = np.concatenate([x_left, x_right])
y = np.concatenate([y_left, y_right])

speeds = np.random.uniform(0.1, 0.2, (N_total, 2))
angle = np.random.uniform(0, 2 * np.pi, N_total)
vx = speeds[:, 0] * np.cos(angle)
vy = speeds[:, 1] * np.sin(angle)

if is_same_gas:
    colors = ['green'] * N_total #Одинаковый газ
else:
    colors = ['blue'] * N_half + ['red'] * N_half #Разные газы


fig, (ax_box, ax_entropy) = plt.subplots(1, 2, figsize=(12, 5))
fig.canvas.manager.set_window_title('Парадокс Гиббса Симуляция')

#Сосуд
ax_box.set_xlim(0, L_WIDTH)
ax_box.set_ylim(0, L_HEIGHT)
ax_box.set_aspect('equal')
ax_box.set_title("Сосуд с газом")

marker_size = (PARTICLE_RADIUS * 40) ** 2 
scatter = ax_box.scatter(x, y, c=colors, s=marker_size, edgecolors='black', zorder=3)

#Отрисовка перегородки
partition, = ax_box.plot([5.0, 5.0], [0, L_HEIGHT], color='black', linewidth=3, linestyle='-', zorder=2)

#Энтропия
ax_entropy.set_xlim(0, 300)
#Динамически настроим высоту графика энтропии
max_possible_entropy = np.log(factorial(N_total) / (factorial(N_half) ** 2)) + 2
ax_entropy.set_ylim(-1, max_possible_entropy if not use_gibbs_correction else max_possible_entropy + 2)
ax_entropy.set_title("Изменение энтропии ($S$)")
ax_entropy.set_xlabel("Кадр (Время)")
ax_entropy.set_ylabel("Энтропия")
ax_entropy.grid(True)

entropy_history = []
time_history = []
entropy_line, = ax_entropy.plot([], [], color='purple', linewidth=2)

def calculate_entropy(x_coords):
    #Сколько частиц сейчас слева (x < 5) и справа (x >= 5)
    n_L = np.sum(x_coords < 5.0)
    n_R = N_total - n_L
       
    n_L = max(0, min(n_L, N_total))
    n_R = max(0, min(n_R, N_total))
    
    #Формула Больцмана
    W = factorial(N_total) / (factorial(n_L) * factorial(n_R))
    S = np.log(W)
    
    if is_same_gas and use_gibbs_correction:
        #Вычтем ложный прирост, чтобы показать константу
        S_mixing_false = np.log(factorial(N_total) / (factorial(N_half) ** 2))
        if n_L != N_half: # когда пошло перемешивание
            #Корректируем значение, чтобы оно не росло ложно
            S = S - (S - np.log(factorial(N_total) / (factorial(N_half)**2)))
            
    return max(0.0, S)

def update(frame):
    global x, y, vx, vy
    
    x += vx
    y += vy
    
    for i in range(N_total):
        for j in range(i + 1, N_total):
            dx = x[j] - x[i]
            dy = y[j] - y[i]
            dist = np.hypot(dx, dy)
            min_dist = PARTICLE_RADIUS * 2
            
            if dist < min_dist:
                #Нормализуем вектор расстояния между центрами
                nx = dx / dist
                ny = dy / dist
                
                #Относительная скорость
                kx = vx[i] - vx[j]
                ky = vy[i] - vy[j]
                
                #Проекция относительной скорости на нормаль
                p = kx * nx + ky * ny
                
                if p > 0: #Движутся навстречу друг другу
                    #Импульс упругого удара
                    vx[i] -= p * nx
                    vy[i] -= p * ny
                    vx[j] += p * nx
                    vy[j] += p * ny
                    
                    overlap = min_dist - dist
                    x[i] -= overlap * 0.5 * nx
                    y[i] -= overlap * 0.5 * ny
                    x[j] += overlap * 0.5 * nx
                    y[j] += overlap * 0.5 * ny
    
    has_partition = frame < WALL_FRAME
    r = PARTICLE_RADIUS
    
    if has_partition:
        partition.set_alpha(1.0) 
        #Корректный отскок от перегородки (х = 5.0) по координатам
        left_hit = (x < 5.0) & (x >= 5.0 - r) & (vx > 0)
        vx[left_hit] *= -1
        right_hit = (x > 5.0) & (x <= 5.0 + r) & (vx < 0)
        vx[right_hit] *= -1
    else:
        partition.set_alpha(0.1) 
        partition.set_linestyle('--')
        
    #Отскок от внешних стенок сосуда с учетом радиуса r
    vx[x <= r] = np.abs(vx[x <= r])
    vx[x >= L_WIDTH - r] = -np.abs(vx[x >= L_WIDTH - r])
    vy[y <= r] = np.abs(vy[y <= r])
    vy[y >= L_HEIGHT - r] = -np.abs(vy[y >= L_HEIGHT - r])
    
    scatter.set_offsets(np.c_[x, y])
    
    current_entropy = calculate_entropy(x)
    entropy_history.append(current_entropy)
    time_history.append(frame)
    entropy_line.set_data(time_history, entropy_history)
    
    return scatter, partition, entropy_line

#Запуск
ani = animation.FuncAnimation(fig, update, frames=300, interval=30, blit=True, repeat=False)
plt.tight_layout()
plt.show()