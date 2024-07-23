from utils import Move


class Board:
    def __init__(self, size):
        self.player_1_loc = [0, size/2]
        self.player_2_loc = [size - 1, size/2]
        self.walls = [[0 for _ in range(size - 1)] for _ in range(size - 1)]
        self.size = size

    def add_wall(self, wall):
        if check_wall_leigitmate(wall):
            self.walls[wall[0][0]][wall[0][1]] = 1
            self.walls[wall[1][0]][wall[1][1]] = 1

    def move_player(self, player_num, move):
        player_loc = self.player_1_loc if player_num == 1 else self.player_2_loc
        self.__check_legal_move(player_loc, move)


    def __check_bounds(self, player_loc, move):
        if player_loc[0] + move[0] < 0 or player_loc[0] + move[0] >= self.size:
            raise Exception("Invalid movement")
        if player_loc[1] + move[1] < 0 or player_loc[1] + move[1] >= self.size:
            raise Exception("Invalid movement")

    def __check_legal_move(self, player_loc, move):
        pass