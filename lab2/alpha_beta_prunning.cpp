#include <iostream>
#include <algorithm>
#include <vector>
#include <array>
using namespace std;

#define WIN 999
#define DRAW 0
#define LOSS -999

#define AI_MARKER 'X'
#define PLAYER_MARKER 'O'
#define EMPTY_SPACE '-'

#define START_DEPTH 0

//打印游戏结果
void print_game_state(int state)
{
	if (state == WIN)
	{
		cout << "恭喜你战胜AI！" << endl;
	}
	else if (state == DRAW)
	{
		cout << "你和AI打成了平手." << endl;
	}
	else if (state == LOSS)
	{
		cout << "你输了，再来一局试试看！" << endl;
	}
}

// 定义所有可能获胜的方式
std::vector<std::vector<std::pair<int, int>>> winning_states{
	// 按行
	{std::make_pair(0, 0), std::make_pair(0, 1), std::make_pair(0, 2)},
	{std::make_pair(1, 0), std::make_pair(1, 1), std::make_pair(1, 2)},
	{std::make_pair(2, 0), std::make_pair(2, 1), std::make_pair(2, 2)},

	// 按列
	{std::make_pair(0, 0), std::make_pair(1, 0), std::make_pair(2, 0)},
	{std::make_pair(0, 1), std::make_pair(1, 1), std::make_pair(2, 1)},
	{std::make_pair(0, 2), std::make_pair(1, 2), std::make_pair(2, 2)},

	// 按对角线
	{std::make_pair(0, 0), std::make_pair(1, 1), std::make_pair(2, 2)},
	{std::make_pair(2, 0), std::make_pair(1, 1), std::make_pair(0, 2)}

};

//打印当前棋盘
void print_board(char board[3][3])
{
	cout << "当前棋盘是：" << endl;
	cout << board[0][0] << " | " << board[0][1] << " | " << board[0][2] << endl;
	cout << "----------" << endl;
	cout << board[1][0] << " | " << board[1][1] << " | " << board[1][2] << endl;
	cout << "----------" << endl;
	cout << board[2][0] << " | " << board[2][1] << " | " << board[2][2] << endl
		 << endl;
}

// 获取所有可用的合法移动
std::vector<std::pair<int, int>> get_legal_moves(char board[3][3])
{
	std::vector<std::pair<int, int>> legal_moves;
	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			if (board[i][j] != AI_MARKER && board[i][j] != PLAYER_MARKER)
			{
				legal_moves.push_back(std::make_pair(i, j));
			}
		}
	}

	return legal_moves;
}

// 判断该位置能否使用
bool position_occupied(char board[3][3], std::pair<int, int> pos)
{
	std::vector<std::pair<int, int>> legal_moves = get_legal_moves(board);

	for (int i = 0; i < legal_moves.size(); i++)
	{
		if (pos.first == legal_moves[i].first && pos.second == legal_moves[i].second)
		{
			return false;
		}
	}

	return true;
}

// 获取给定一方，所有其所下的棋位置
std::vector<std::pair<int, int>> get_occupied_positions(char board[3][3], char marker)
{
	std::vector<std::pair<int, int>> occupied_positions;

	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			if (marker == board[i][j])
			{
				occupied_positions.push_back(std::make_pair(i, j));
			}
		}
	}

	return occupied_positions;
}

// 查看棋盘是否为空
bool board_is_full(char board[3][3])
{
	std::vector<std::pair<int, int>> legal_moves = get_legal_moves(board);

	if (0 == legal_moves.size())
	{
		return true;
	}
	else
	{
		return false;
	}
}

// 判断该方赢了吗
bool game_is_won(std::vector<std::pair<int, int>> occupied_positions)
{
	bool game_won;

	for (int i = 0; i < winning_states.size(); i++)
	{
		game_won = true;
		std::vector<std::pair<int, int>> curr_win_state = winning_states[i];
		for (int j = 0; j < 3; j++)
		{
			if (!(std::find(std::begin(occupied_positions), std::end(occupied_positions), curr_win_state[j]) != std::end(occupied_positions)))
			{
				game_won = false;
				break;
			}
		}

		if (game_won)
		{
			break;
		}
	}
	return game_won;
}

// 切换下棋方
char get_opponent_marker(char marker)
{
	char opponent_marker;
	if (marker == PLAYER_MARKER)
		opponent_marker = AI_MARKER;
	else
		opponent_marker = PLAYER_MARKER;

	return opponent_marker;
}

// 判断谁赢谁输
int get_board_state(char board[3][3], char marker)
{

	char opponent_marker = get_opponent_marker(marker);

	std::vector<std::pair<int, int>> occupied_positions = get_occupied_positions(board, marker);

	bool is_won = game_is_won(occupied_positions);

	if (is_won)
	{
		return WIN;
	}

	occupied_positions = get_occupied_positions(board, opponent_marker);
	bool is_lost = game_is_won(occupied_positions);

	if (is_lost)
	{
		return LOSS;
	}

	bool is_full = board_is_full(board);
	if (is_full)
	{
		return DRAW;
	}

	return DRAW;
}

// 使用alpha-beta剪枝法，进行剪枝，确定AI下棋的位置
std::pair<int, std::pair<int, int>> minimax_optimization(char board[3][3], char marker, int depth, int alpha, int beta)
{
	// Initialize best move
	std::pair<int, int> best_move = std::make_pair(-1, -1);
	int best_score = (marker == AI_MARKER) ? LOSS : WIN;

	// If we hit a terminal state (leaf node), return the best score and move
	if (board_is_full(board) || DRAW != get_board_state(board, AI_MARKER))
	{
		best_score = get_board_state(board, AI_MARKER);
		return std::make_pair(best_score, best_move);
	}

	std::vector<std::pair<int, int>> legal_moves = get_legal_moves(board);

	for (int i = 0; i < legal_moves.size(); i++)
	{
		std::pair<int, int> curr_move = legal_moves[i];
		board[curr_move.first][curr_move.second] = marker;

		// Maximizing player's turn
		if (marker == AI_MARKER)
		{
			int score = minimax_optimization(board, PLAYER_MARKER, depth + 1, alpha, beta).first;

			// Get the best scoring move
			if (best_score < score)
			{
				best_score = score - depth * 10;
				best_move = curr_move;

				// Check if this branch's best move is worse than the best
				// option of a previously search branch. If it is, skip it
				alpha = std::max(alpha, best_score);
				board[curr_move.first][curr_move.second] = EMPTY_SPACE;
				if (beta <= alpha)
				{
					break;
				}
			}

		} // Minimizing opponent's turn
		else
		{
			int score = minimax_optimization(board, AI_MARKER, depth + 1, alpha, beta).first;

			if (best_score > score)
			{
				best_score = score + depth * 10;
				best_move = curr_move;

				// Check if this branch's best move is worse than the best
				// option of a previously search branch. If it is, skip it
				beta = std::min(beta, best_score);
				board[curr_move.first][curr_move.second] = EMPTY_SPACE;
				if (beta <= alpha)
				{
					break;
				}
			}
		}

		board[curr_move.first][curr_move.second] = EMPTY_SPACE; // Undo move
	}

	return std::make_pair(best_score, best_move);
}

// 判断游戏是否结束
bool game_is_done(char board[3][3])
{
	if (board_is_full(board))
		return true;

	if (DRAW != get_board_state(board, AI_MARKER))

		return true;
	return false;
}

int main()
{
	//初始化棋盘
	char board[3][3] = {{EMPTY_SPACE, EMPTY_SPACE, EMPTY_SPACE}, {EMPTY_SPACE, EMPTY_SPACE, EMPTY_SPACE}, {EMPTY_SPACE, EMPTY_SPACE, EMPTY_SPACE}};

	cout << "*****************和AI下一字棋*****************" << endl
		 << endl;
	cout << "用'O'表示游戏方 \t 用'X'表示AI" << endl
		 << endl;

	print_board(board);

	//主循环，先判断一下游戏是否结束
	while (!game_is_done(board))
	{
		int row, col;
		cout << "请选择你要下的位置: ";
		cin >> row >> col;
		cout << endl
			 << endl;

		//如果这个位置被占了，让player重下
		if (position_occupied(board, std::make_pair(row, col)))
		{
			cout << "(" << row << ", " << col << ") 位置 不能下棋，请换一个位置下棋吧" << endl;
			continue;
		}
		else
		{
			board[row][col] = PLAYER_MARKER;
		}

		//AI下棋
		std::pair<int, std::pair<int, int>> ai_move = minimax_optimization(board, AI_MARKER, START_DEPTH, LOSS, WIN);
		board[ai_move.second.first][ai_move.second.second] = AI_MARKER;

		print_board(board);
	}

	cout << "********** 游戏结束 **********" << endl
		 << endl;

	//输出player是否战胜AI
	int player_state = get_board_state(board, PLAYER_MARKER);
	print_game_state(player_state);

	return 0;
}