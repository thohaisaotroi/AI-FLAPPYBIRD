import pygame , sys , random , neat , os
import pickle
import matplotlib.pyplot as plt

pygame.init()

SCREEN_WIDTH , SCREEN_HEIGHT = 576 , 800
FPS = 60

screen = pygame.display.set_mode((SCREEN_WIDTH,SCREEN_HEIGHT))
pygame.display.set_caption("Flappy Bird")
clock = pygame.time.Clock()

BIRD_IMAGES = [pygame.image.load(f"imgs/bird{i+1}.png").convert_alpha() for i in range(3)]
PIPE_IMG =  pygame.image.load("imgs/pipe.png").convert_alpha()
BG_IMG = pygame.transform.scale2x(pygame.image.load("imgs/bg.png").convert_alpha())
BASE_IMG =  pygame.image.load("imgs/base.png").convert_alpha()

pygame.display.set_icon(BIRD_IMAGES[1])

font = pygame.font.Font(pygame.font.get_default_font(), 30)

class Base:
    def __init__(self , x):
        self.base_img = BASE_IMG
        self.rect = BASE_IMG.get_rect()
        self.rect.x = x
        self.rect.y = SCREEN_HEIGHT - self.rect.height

    def move_base(self):
        self.rect.x -= 1
        if self.rect.right <= 0:
            self.rect.x = self.rect.width * 2

    def draw(self):
        screen.blit(self.base_img,self.rect)

    def update(self):
        self.move_base()
        self.draw()

class Bird:
    def __init__(self):
        self.bird_img = BIRD_IMAGES[0]
        self.rect = self.bird_img.get_rect()
        self.rect.x = 100
        self.rect.y = SCREEN_HEIGHT//2
        self.max_fall_speed = 10
        self.gravity = 0.5
        self.y_vel = 0
        self.flap = False
        self.frame = 0

    def Gravity(self):
        if self.rect.y < SCREEN_HEIGHT - 136:
            self.y_vel += self.gravity
            if self.y_vel >= self.max_fall_speed:
                self.y_vel = self.max_fall_speed

        if self.y_vel >0:
            self.flap = False

        self.rect.y += self.y_vel * 0.9

        if self.rect.y >= SCREEN_HEIGHT - 136:
            self.rect.y = SCREEN_HEIGHT - 136
            self.y_vel = 0

    def animation(self):
        if self.frame < FPS//6:
            self.bird_img = BIRD_IMAGES[0]
        elif FPS//6<= self.frame < FPS//3:
            self.bird_img = BIRD_IMAGES[1]
        else:
            self.bird_img = BIRD_IMAGES[2]

        if self.frame > (FPS//3)*2 :
            self.frame = 0
        self.frame += 1

    def draw_bird(self):
        angle = min(max(self.y_vel * -5, -25), 25)
        rotated_bird = pygame.transform.rotate(self.bird_img, angle)
        rotated_rect = rotated_bird.get_rect(center=self.rect.center)
        screen.blit(rotated_bird, rotated_rect)

    def update_bird(self):
        self.animation()
        self.Gravity()
        self.draw_bird()

class Pipe():
    def __init__(self , x , y, speed_x=3):
        self.pipe_img = PIPE_IMG
        self.rotated_pipe_img = pygame.transform.rotate(PIPE_IMG,180)
        self.rect = self.pipe_img.get_rect()
        self.rotated_rect = self.rotated_pipe_img.get_rect()
        self.gap = 140
        self.rect.x = x
        self.rotated_rect.x = x
        self.rect.y = y
        self.rotated_rect.bottom = y - self.gap
        self.passed = False
        self.direction = random.choice([-1, 1])
        self.speed_y = 1
        self.speed_x = speed_x

    def move_pipe(self):
        self.rect.x -= self.speed_x
        self.rotated_rect.x -= self.speed_x
        self.rect.y += self.speed_y * self.direction
        self.rotated_rect.bottom = self.rect.y - self.gap

        if self.rect.y > 600 or self.rect.y < 200:
            self.direction *= -1

    def draw_pipe(self):
        screen.blit(self.pipe_img,self.rect)
        screen.blit(self.rotated_pipe_img,self.rotated_rect)

    def update_pipe(self):
        self.draw_pipe()
        self.move_pipe()

base_list = [Base(0) , Base(336),Base(672)]
pipe_list = []

def fitness(genomes, config):
    global score
    score = 0
    birds = []
    nets = []
    genomes_list = []
    pipe_list = []

    base_speed = 3.0
    PIPE_GAP_DISTANCE = 350

    pipe_speed = base_speed
    pipe_timer = 0
    pipe_interval = int(PIPE_GAP_DISTANCE / pipe_speed)

    for _, genome in genomes:
        birds.append(Bird())
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        genomes_list.append(genome)
        genome.fitness = 0

    y = random.randint(240, 588)
    pipe_list.append(Pipe(SCREEN_WIDTH, y, pipe_speed))

    while len(birds) > 0:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        screen.blit(BG_IMG, (0, -100))

        # Cộng điểm sống sót cho từng genome
        for genome in genomes_list:
            genome.fitness += 0.1

        prev_pipe_speed = pipe_speed
        pipe_speed = base_speed * (1.1 ** (score // 10))
        pipe_interval = int(PIPE_GAP_DISTANCE / pipe_speed)
        if pipe_speed != prev_pipe_speed:
            pipe_timer = 0

        pipe_timer += 1
        if pipe_timer >= pipe_interval:
            y = random.randint(240, 588)
            pipe_list.append(Pipe(SCREEN_WIDTH, y, pipe_speed))
            pipe_timer = 0

        for pipe in pipe_list[:]:
            pipe.update_pipe()
            if pipe.rect.x < -60:
                pipe_list.remove(pipe)

        for base in base_list:
            base.update()

        birds_to_remove = []
        for i, (bird, genome, net) in enumerate(zip(birds, genomes_list, nets)):
            closest_pipe = None
            for pipe in pipe_list:
                if pipe.rect.x + pipe.rect.width > bird.rect.x:
                    closest_pipe = pipe
                    break

            if closest_pipe:
                bird_y = bird.rect.y
                bird_velocity = bird.y_vel
                bottom_gap_y = closest_pipe.rect.y
                top_gap_y = closest_pipe.rotated_rect.bottom
                distance_to_pipe = closest_pipe.rect.x - bird.rect.x

                inputs = [
                    bird_y,
                    bird_velocity,
                    distance_to_pipe,
                    bird_y - top_gap_y,
                    bottom_gap_y - bird_y,
                ]

                output = net.activate(inputs)
                if output[0] > 0.5:
                    bird.y_vel = -8
                    bird.flap = True

            bird.update_bird()

            for pipe in pipe_list:
                if bird.rect.colliderect(pipe.rect) or bird.rect.colliderect(pipe.rotated_rect) \
                   or bird.rect.bottom >= SCREEN_HEIGHT - 136 or bird.rect.y < 0:
                    genome.fitness -= 1
                    birds_to_remove.append(i)

                if not pipe.passed and pipe.rect.x < bird.rect.x:
                    pipe.passed = True
                    score += 1
                    genome.fitness += 5

        for i in sorted(set(birds_to_remove), reverse=True):
            if i < len(birds):
                birds.pop(i)
            if i < len(nets):
                nets.pop(i)
            if i < len(genomes_list):
                genomes_list.pop(i)

        score_text = font.render(f"Score: {score}", True, (255, 255, 255))
        screen.blit(score_text, (20, 20))
        pygame.display.flip()
        clock.tick(FPS)



def save_best_genome(genome, filename='best_genome.pkl'):
    with open(filename, 'wb') as f:
        pickle.dump(genome, f)

def plot_fitness(stats, filename='fitness_plot.png'):
    generation = range(len(stats.most_fit_genomes))
    best_fitness = [g.fitness for g in stats.most_fit_genomes]
    avg_fitness = stats.get_fitness_mean()
    stdev_fitness = stats.get_fitness_stdev()

    plt.figure(figsize=(10, 5))
    plt.plot(generation, best_fitness, label='Best Fitness')
    plt.plot(generation, avg_fitness, label='Average Fitness')
    plt.fill_between(generation,
                     [a - s for a, s in zip(avg_fitness, stdev_fitness)],
                     [a + s for a, s in zip(avg_fitness, stdev_fitness)],
                     alpha=0.2, label='Std Dev')

    plt.title('Fitness over Generations')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def run_neat(config_file):
    # Đọc file cấu hình NEAT
    config = neat.config.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_file
    )

    # Tạo quần thể NEAT
    population = neat.Population(config)

    # Thêm các báo cáo theo dõi
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)

    # Chạy thuật toán tiến hóa trong 15 thế hệ
    winner = population.run(fitness, 15)

    # Lưu genome tốt nhất
    save_best_genome(winner, 'best_genome.pkl')

    #  Vẽ biểu đồ quá trình huấn luyện
    plot_fitness(stats, 'fitness_plot.png')

    print(" Đã hoàn tất huấn luyện.")
    print(" Genome tốt nhất được lưu tại: best_genome.pkl")
    print(" Biểu đồ quá trình huấn luyện được lưu tại: fitness_plot.png")


    

if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config.txt')
    run_neat(config_path)
