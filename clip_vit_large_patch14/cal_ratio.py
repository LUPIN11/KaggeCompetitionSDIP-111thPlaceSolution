from sympy import symbols, solve

# # 定义未知数
# x, y = symbols('x y')
#
# # 定义方程组
# eq1 = 2*x + 3*y - 6
# eq2 = x - y - 2
#
# # 解方程组
# sol = solve((eq1, eq2), (x, y))
#
# # 输出解
# print(sol)


A, B, C = symbols('A B C')

cosa = 0.7041
cosb = 0.6988
cosc = 0.7636

eq1 = A + B*cosa*cosb + C*cosa*cosc - cosa
eq2 = A*cosa*cosb + B + C*cosb*cosc - cosb
eq3 = A*cosa*cosc + B*cosb*cosc + C - cosc

sol = solve((eq1, eq2, eq3), (A, B, C))

print(sol)