import pygame
import pickle
import neat
import os
import random
from main import Bird, Pipe, Base, SCREEN_WIDTH, SCREEN_HEIGHT, BG_IMG, font, FPS

def load_genome(config_path):
    with open("best_genome.pkl", "rb") as f:
        genome = pickle.load(f)
    config = neat.config.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path
    )
    return neat.nn.FeedForwardNetwork.create(genome, config)

def run_ai_test(net, title):
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption(title)
    clock = pygame.time.Clock()

    bird = Bird()
    pipes = []
    base_list = [Base(0), Base(336), Base(672)]
    score = 0
    pipe_speed = 3.0
    PIPE_GAP_DISTANCE = 350
    pipe_timer = 0
    pipe_interval = int(PIPE_GAP_DISTANCE / pipe_speed)

    y = random.randint(240, 588)
    pipes.append(Pipe(SCREEN_WIDTH, y, pipe_speed))
    running = True
    while running:
        screen.blit(BG_IMG, (0, -100))
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        prev_pipe_speed = pipe_speed
        pipe_speed = 3.0 * (1.2 ** (score // 10))
        pipe_interval = int(PIPE_GAP_DISTANCE / pipe_speed)
        if pipe_speed != prev_pipe_speed:
            pipe_timer = 0

        pipe_timer += 1
        if pipe_timer >= pipe_interval:
            y = random.randint(240, 588)
            pipes.append(Pipe(SCREEN_WIDTH, y, pipe_speed))
            pipe_timer = 0

        for pipe in pipes[:]:
            pipe.update_pipe()
            if pipe.rect.x < -60:
                pipes.remove(pipe)

        for base in base_list:
            base.update()

        closest_pipe = None
        for pipe in pipes:
            if pipe.rect.x + pipe.rect.width > bird.rect.x:
                closest_pipe = pipe
                break

        if closest_pipe:
            inputs = [
                bird.rect.y,
                bird.y_vel,
                closest_pipe.rect.x - bird.rect.x,
                bird.rect.y - closest_pipe.rotated_rect.bottom,
                closest_pipe.rect.y - bird.rect.y,
            ]
            output = net.activate(inputs)
            if output[0] > 0.5:
                bird.y_vel = -8
                bird.flap = True

        bird.update_bird()

        for pipe in pipes:
            if bird.rect.colliderect(pipe.rect) or bird.rect.colliderect(pipe.rotated_rect) or \
               bird.rect.bottom >= SCREEN_HEIGHT - 136 or bird.rect.y < 0:
                print(f"{title} kết thúc! Score = {score}")
                running = False

            if not pipe.passed and pipe.rect.x < bird.rect.x:
                pipe.passed = True
                score += 1

        score_text = font.render(f"Score: {score}", True, (255, 255, 255))
        screen.blit(score_text, (20, 20))
        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()

if __name__ == "__main__":
    config_path = os.path.join(os.path.dirname(__file__), "config.txt")
    print(f"\n Testing best_genome.pkl...")
    net = load_genome(config_path)
    run_ai_test(net, "Best Genome")
