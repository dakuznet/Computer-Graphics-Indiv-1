import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection
from typing import List, Tuple, Optional

Point = Tuple[float, float]


# ═══════════════════════════════════════════════════════════
# Геометрические предикаты
# ═══════════════════════════════════════════════════════════

def cross(o: Point, a: Point, b: Point) -> float:
    """Векторное произведение OA × OB. Знак показывает ориентацию."""
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def orient_ccw(poly: List[Point]) -> List[Point]:
    """Привести полигон к порядку против часовой стрелки."""
    area = 0.0
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    if area < 0:
        return poly[::-1]
    return poly


def is_reflex(poly: List[Point], i: int) -> bool:
    """
    Является ли вершина i рефлексной.
    """
    n = len(poly)
    prev_p = poly[(i - 1) % n]
    curr_p = poly[i]
    next_p = poly[(i + 1) % n]
    return cross(prev_p, curr_p, next_p) < 0


def point_in_polygon(p: Point, poly: List[Point]) -> bool:
    """Принадлежность точки полигону."""
    x, y = p
    n = len(poly)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if (yi > y) != (yj > y):
            x_int = (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi
            if x < x_int:
                inside = not inside
        j = i
    return inside


def segments_properly_intersect(a1: Point, a2: Point,
                                 b1: Point, b2: Point) -> bool:
    """Строгое пересечение отрезков."""
    d1 = cross(b1, b2, a1)
    d2 = cross(b1, b2, a2)
    d3 = cross(a1, a2, b1)
    d4 = cross(a1, a2, b2)
    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True
    return False


def _on_segment(a: Point, b: Point, p: Point) -> bool:
    """Лежит ли точка p на отрезке a-b."""
    return (min(a[0], b[0]) - 1e-9 <= p[0] <= max(a[0], b[0]) + 1e-9 and
            min(a[1], b[1]) - 1e-9 <= p[1] <= max(a[1], b[1]) + 1e-9)


def _point_in_triangle_strict(p: Point, a: Point, b: Point, c: Point) -> bool:
    """Строго внутри треугольника abc."""
    d1 = cross(a, b, p)
    d2 = cross(b, c, p)
    d3 = cross(c, a, p)
    has_neg = (d1 < -1e-9) or (d2 < -1e-9) or (d3 < -1e-9)
    has_pos = (d1 >  1e-9) or (d2 >  1e-9) or (d3 >  1e-9)
    return not (has_neg and has_pos)


def diagonal_inside(poly: List[Point], i: int, j: int) -> bool:
    """
    Лежит ли диагональ poly[i]-poly[j] строго внутри полигона.
    (A) ни одна другая вершина не лежит на диагонали;
    (B) диагональ не пересекает рёбра (кроме инцидентных концам);
    (C) середина диагонали внутри полигона;
    (D) диагональ попадает во внутренний угол в обеих вершинах.
    """
    n = len(poly)
    if abs(i - j) == 1 or abs(i - j) == n - 1:
        return False

    vi, vj = poly[i], poly[j]

    # (A)
    for m in range(n):
        if m == i or m == j:
            continue
        if abs(cross(vi, vj, poly[m])) < 1e-9 and _on_segment(vi, vj, poly[m]):
            return False

    # (B)
    for k in range(n):
        a = poly[k]
        b = poly[(k + 1) % n]
        if k == i or k == j or (k + 1) % n == i or (k + 1) % n == j:
            continue
        if segments_properly_intersect(vi, vj, a, b):
            return False

    # (C)
    mid = ((vi[0] + vj[0]) / 2.0, (vi[1] + vj[1]) / 2.0)
    if not point_in_polygon(mid, poly):
        return False

    # (D)
    for a, b in ((i, j), (j, i)):
        va, vb = poly[a], poly[b]
        prev_a = poly[(a - 1) % n]
        next_a = poly[(a + 1) % n]
        c_prev = cross(va, prev_a, vb)
        c_next = cross(va, next_a, vb)
        if is_reflex(poly, a):
            if c_prev > 0 and c_next < 0:
                return False
        else:
            if c_next <= 0 or c_prev >= 0:
                return False

    return True


def find_valid_ear(poly: List[Point]
                   ) -> Optional[Tuple[Tuple[Point, Point, Point], List[Point]]]:
    """
    Ищет валидное ухо: тройка подряд идущих вершин (a,b,c),
    у которой хорда a-c — внутренняя диагональ, и внутри abc нет других вершин.
    """
    n = len(poly)
    for k in range(n):
        a = poly[k]
        b = poly[(k + 1) % n]
        c = poly[(k + 2) % n]
        if not diagonal_inside(poly, k, (k + 2) % n):
            continue
        ok = True
        for m in range(n):
            if m == k or m == (k + 1) % n or m == (k + 2) % n:
                continue
            if _point_in_triangle_strict(poly[m], a, b, c):
                ok = False
                break
        if ok:
            skip = (k + 1) % n
            rest = [poly[m] for m in range(n) if m != skip]
            return ((a, b, c), rest)
    return None


# ═══════════════════════════════════════════════════════════
# Поиск вторгающейся вершины
# ═══════════════════════════════════════════════════════════

def find_intruding_pair(poly: List[Point]) -> Optional[Tuple[int, int]]:
    """
    Найти пару (i, j): i — рефлексная вершина, j — видимая из i.
    Возвращает None, если такой пары нет.
    """
    n = len(poly)
    for i in range(n):
        if not is_reflex(poly, i):
            continue
        for offset in range(2, n - 1):
            j = (i + offset) % n
            if diagonal_inside(poly, i, j):
                return (i, j)
    return None


def find_intrusion_on_edge(poly: List[Point], reflex_idx: int
                            ) -> Optional[Tuple[int, Point]]:
    """
    Если видимой вершины нет — вторгаемся в ребро.
    Строим луч из рефлексной вершины в направлении биссектрисы
    внутреннего угла и находим первое пересечение с ребром полигона.
    Возвращает (idx_ребра, точка_вторжения) или None.
    """
    n = len(poly)
    R = poly[reflex_idx]
    prev_p = poly[(reflex_idx - 1) % n]
    next_p = poly[(reflex_idx + 1) % n]

    v1 = (prev_p[0] - R[0], prev_p[1] - R[1])
    v2 = (next_p[0] - R[0], next_p[1] - R[1])

    len1 = np.hypot(*v1)
    len2 = np.hypot(*v2)
    if len1 < 1e-12 or len2 < 1e-12:
        return None
    u1 = (v1[0] / len1, v1[1] / len1)
    u2 = (v2[0] / len2, v2[1] / len2)

    bis = (u1[0] + u2[0], u1[1] + u2[1])
    blen = np.hypot(*bis)
    if blen < 1e-12:
        return None
    bis = (bis[0] / blen, bis[1] / blen)

    best_t = float('inf')
    best_edge = -1
    best_point = None

    for k in range(n):
        a = poly[k]
        b = poly[(k + 1) % n]
        if k == reflex_idx or (k + 1) % n == reflex_idx:
            continue

        dx = b[0] - a[0]
        dy = b[1] - a[1]
        denom = bis[0] * dy - bis[1] * dx
        if abs(denom) < 1e-12:
            continue
        rx = a[0] - R[0]
        ry = a[1] - R[1]
        t = (rx * dy - ry * dx) / denom
        s = (rx * bis[1] - ry * bis[0]) / denom
        if t > 1e-9 and 0.0 <= s <= 1.0 and t < best_t:
            best_t = t
            best_edge = k
            best_point = (R[0] + t * bis[0], R[1] + t * bis[1])

    if best_point is not None:
        return (best_edge, best_point)
    return None


# ═══════════════════════════════════════════════════════════
# Основной алгоритм триангуляции
# ═══════════════════════════════════════════════════════════

def triangulate_intruding(poly: List[Point],
                          steps: Optional[list] = None
                          ) -> List[Tuple[Point, Point, Point]]:
    """
    Триангуляция методом вторгающихся вершин.
    
    Аргументы:
        poly  — список вершин полигона
        steps — если не None, сюда добавляются шаги для визуализации
    
    Возвращает:
        Список треугольников [(A, B, C), ...]
    """
    poly = orient_ccw(list(poly))

    # Базовый случай
    if len(poly) == 3:
        if steps is not None:
            steps.append({'type': 'base', 'poly': poly[:]})
        return [tuple(poly)]

    if len(poly) < 3:
        return []

    # Шаг 1: ищем рефлексную вершину с видимой парой
    pair = find_intruding_pair(poly)

    if pair is not None:
        i, j = pair
        vi, vj = poly[i], poly[j]

        if i < j:
            poly1 = poly[i:j + 1]
            poly2 = poly[j:] + poly[:i + 1]
        else:
            poly1 = poly[j:i + 1]
            poly2 = poly[i:] + poly[:j + 1]

        if steps is not None:
            steps.append({
                'type': 'diagonal',
                'poly': poly[:],
                'reflex': vi,
                'visible': vj,
                'poly1': poly1[:],
                'poly2': poly2[:],
            })

        return (triangulate_intruding(poly1, steps) +
                triangulate_intruding(poly2, steps))

    # Шаг 2: видимой вершины нет — вторгаемся в ребро
    # Находим любую рефлексную вершину
    reflex_idx = -1
    for k in range(len(poly)):
        if is_reflex(poly, k):
            reflex_idx = k
            break

    if reflex_idx == -1:
        # Полигон выпуклый — ищем валидное ухо (гарантированно существует)
        ear_result = find_valid_ear(poly)
        if ear_result is not None:
            tri, rest = ear_result
            if steps is not None:
                steps.append({'type': 'convex_ear', 'poly': poly[:], 'ear': tri})
            return [tri] + triangulate_intruding(rest, steps)
        tri = (poly[0], poly[1], poly[2])
        rest = [poly[0], poly[2]] + poly[3:]
        if steps is not None:
            steps.append({'type': 'fallback_ear', 'poly': poly[:], 'ear': tri})
        return [tri] + triangulate_intruding(rest, steps)

    intrusion = find_intrusion_on_edge(poly, reflex_idx)

    if intrusion is not None:
        edge_idx, new_point = intrusion
        new_poly = poly[:edge_idx + 1] + [new_point] + poly[edge_idx + 1:]
        new_reflex_idx = reflex_idx if reflex_idx <= edge_idx else reflex_idx + 1
        new_visible_idx = edge_idx + 1

        i, j = new_reflex_idx, new_visible_idx
        vi, vj = new_poly[i], new_poly[j]

        if i < j:
            poly1 = new_poly[i:j + 1]
            poly2 = new_poly[j:] + new_poly[:i + 1]
        else:
            poly1 = new_poly[j:i + 1]
            poly2 = new_poly[i:] + new_poly[:j + 1]

        if steps is not None:
            steps.append({
                'type': 'intrusion',
                'poly': poly[:],
                'reflex': vi,
                'new_point': new_point,
                'poly1': poly1[:],
                'poly2': poly2[:],
            })

        return (triangulate_intruding(poly1, steps) +
                triangulate_intruding(poly2, steps))

    # Шаг 3: ищем валидное ухо
    ear_result = find_valid_ear(poly)
    if ear_result is not None:
        tri, rest = ear_result
        if steps is not None:
            steps.append({'type': 'ear', 'poly': poly[:], 'ear': tri})
        return [tri] + triangulate_intruding(rest, steps)

    tri = (poly[0], poly[1], poly[2])
    rest = [poly[0], poly[2]] + poly[3:]
    if steps is not None:
        steps.append({'type': 'fallback_ear', 'poly': poly[:], 'ear': tri})
    return [tri] + triangulate_intruding(rest, steps)


# ═══════════════════════════════════════════════════════════
# Визуализация
# ═══════════════════════════════════════════════════════════

def draw_poly(ax, poly, color='black', lw=2, alpha=1.0):
    """Нарисовать контур полигона."""
    pts = list(poly) + [poly[0]]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    ax.plot(xs, ys, color=color, lw=lw, alpha=alpha)


def draw_vertices(ax, poly, color='red', size=50):
    """Нарисовать вершины."""
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    ax.scatter(xs, ys, c=color, s=size, zorder=5)


def visualize_triangulation(poly: List[Point], title: str = "Триангуляция"):
    """Полная визуализация: исходный полигон + шаги + результат."""
    steps = []
    triangles = triangulate_intruding(poly, steps)

    n_steps = len(steps)
    n_total = n_steps + 2
    n_cols = 3
    n_rows = (n_total + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4.5 * n_rows))
    axes = np.array(axes).flatten()

    # Шаг 0: исходный полигон
    ax = axes[0]
    ax.set_title("Исходный полигон", fontsize=11)
    draw_poly(ax, poly)
    draw_vertices(ax, poly)
    for i, p in enumerate(poly):
        ax.annotate(f"V{i}", p, textcoords="offset points",
                    xytext=(5, 5), fontsize=8)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # Шаги алгоритма
    for idx, step in enumerate(steps):
        ax = axes[idx + 1]
        ax.set_title(f"Шаг {idx + 1}: {step['type']}", fontsize=10)

        if step['type'] in ('diagonal', 'intrusion'):
            draw_poly(ax, step['poly'])
            draw_vertices(ax, step['poly'])
            if step['type'] == 'diagonal':
                ax.plot([step['reflex'][0], step['visible'][0]],
                        [step['reflex'][1], step['visible'][1]],
                        'g--', lw=2, label='диагональ')
                ax.scatter(*step['reflex'], c='red', s=100, zorder=6,
                           label='рефлексная')
                ax.scatter(*step['visible'], c='blue', s=100, zorder=6,
                           label='видимая')
            else:
                ax.plot([step['reflex'][0], step['new_point'][0]],
                        [step['reflex'][1], step['new_point'][1]],
                        'm--', lw=2, label='луч вторжения')
                ax.scatter(*step['reflex'], c='red', s=100, zorder=6)
                ax.scatter(*step['new_point'], c='purple', s=120,
                           marker='*', zorder=6, label='новая вершина')
            draw_poly(ax, step['poly1'], color='green', lw=1.5, alpha=0.7)
            draw_poly(ax, step['poly2'], color='orange', lw=1.5, alpha=0.7)
            ax.legend(fontsize=7, loc='upper right')

        elif step['type'] in ('base', 'convex_ear', 'fallback_ear', 'ear'):
            if step['type'] == 'base':
                draw_poly(ax, step['poly'], color='green')
                draw_vertices(ax, step['poly'], color='green')
                ax.set_title(f"Шаг {idx+1}: базовый (треугольник)", fontsize=10)
            else:
                draw_poly(ax, step['poly'])
                draw_vertices(ax, step['poly'])
                ear = step['ear']
                tri_pts = list(ear) + [ear[0]]
                xs = [p[0] for p in tri_pts]
                ys = [p[1] for p in tri_pts]
                ax.fill(xs, ys, alpha=0.3, color='yellow')
                ax.plot(xs, ys, 'y-', lw=2)

        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

    # Результат
    ax = axes[n_steps + 1]
    ax.set_title(f"Результат: {len(triangles)} треугольников", fontsize=11)
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(triangles), 1)))
    for i, tri in enumerate(triangles):
        pts = list(tri) + [tri[0]]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.fill(xs, ys, alpha=0.4, color=colors[i % len(colors)])
        ax.plot(xs, ys, 'k-', lw=1)
    draw_poly(ax, poly, lw=2)
    draw_vertices(ax, poly)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    for idx in range(n_steps + 2, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


# ═══════════════════════════════════════════════════════════
# Тестовые примеры
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Пример 1: невыпуклый полигон (L-образный)
    poly_L = [(0, 0), (4, 0), (4, 2), (2, 2), (2, 4), (0, 4)]
    visualize_triangulation(poly_L, "Пример 1: L-образный полигон")

    # Пример 2: звёздообразный полигон
    angles = np.linspace(0, 2 * np.pi, 10, endpoint=False)
    radii = np.array([3, 1.5] * 5)
    poly_star = [(r * np.cos(a), r * np.sin(a)) for a, r in zip(angles, radii)]
    visualize_triangulation(poly_star, "Пример 2: Звезда (5 лучей)")

    # Пример 3: сложный невыпуклый полигон (П-образный)
    poly_complex = [(0, 0), (6, 0), (6, 2), (4, 2), (4, 4),
                    (6, 4), (6, 6), (0, 6), (0, 4), (2, 4), (2, 2), (0, 2)]
    visualize_triangulation(poly_complex, "Пример 3: Сложный полигон (П-образный)")

    # Пример 4: выпуклый полигон
    poly_convex = [(0, 0), (5, 0), (6, 3), (3, 5), (-1, 3)]
    visualize_triangulation(poly_convex, "Пример 4: Выпуклый полигон")

    # Пример 5: квадратная спираль / буква G
    poly_G = [
        (0, 0), (10, 0), (10, 10), (5, 10),
        (5, 5), (7, 5), (7, 7), (9, 7),
        (9, 2), (3, 2), (3, 10), (0, 10),
    ]
    visualize_triangulation(poly_G, "Пример 5: Спираль / буква G")
