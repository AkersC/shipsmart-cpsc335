"""
core.py - THE ALGORITHM ENGINE  (CPSC 335 Final Project)

This is the file you complete. Every project option (GridRestore, ShipSmart,
BloodLink) uses these SAME functions, so the app works as soon as they do.

    Part 1  build_index                      hash table + duplicate check
    Part 2  merge_sort, lower_bound          divide & conquer sort, binary search
    Part 3  MinHeap                          priority queue
    Part 4  greedy_select                    greedy algorithm
    Part 5  dp_select                        dynamic programming (0/1 knapsack)
    Part 6  bfs_reachable, dijkstra, mst_kruskal   graph algorithms
    Part 7  kmp_search                       string matching
    Part 8  tour_brute_force, tour_heuristic an NP-hard problem (TSP)

RULES (checked by check_project.py):
  * Write the algorithms yourself. Inside this file do NOT use:
    sorted(), .sort(), heapq, bisect, str.find(), str.index(), str.count(), re.
  * dict, set, list, collections.deque and itertools.permutations ARE allowed.
  * Do not change any function name or its inputs/outputs. The app and the
    tests depend on them.

Delete each `raise NotImplementedError(...)` line when you write that function.
Run  python check_project.py  to see which parts pass.
"""

from collections import deque
from itertools import permutations
import time


# =====================================================================
# PART 1 - Hash table lookup                              (Weeks 2, 11)
# =====================================================================
def build_index(records, key):
    """Index records by record[key] and find duplicate keys.

    records : list of dicts, e.g. [{"id": "A", ...}, {"id": "B", ...}]
    key     : the field to index by, e.g. "id"

    Return (index, duplicates)
      index      : dict  key value -> record   (keep the FIRST record seen)
      duplicates : list of key values seen more than once, in the order found

    Example: records with ids A, B, A  ->  ({"A": ..., "B": ...}, ["A"])
    Target: expected O(n) time.
    """
    raise NotImplementedError("Part 1: build_index")


# =====================================================================
# PART 2 - Merge sort and binary search                        (Week 4)
# =====================================================================
def merge_sort(items, key=lambda x: x):
    """Return a NEW list with the items sorted by key(item), smallest first.

    Must be STABLE: items with equal keys keep their original order.
    Steps: 1) a list of 0 or 1 items is already sorted
           2) split in half and merge_sort each half (recursion)
           3) merge the two sorted halves: repeatedly take the smaller front item
              (when the keys are equal, take from the LEFT half -> stable)
    Target: O(n log n).
    """
    # Base case: 0 or 1 items is already sorted (return a copy, not the original)
    if len(items) <= 1:
        return list(items)

    # Divide and conquer: sort each half recursively
    mid = len(items) // 2
    left = merge_sort(items[:mid], key)
    right = merge_sort(items[mid:], key)

    # Merge: repeatedly take the smaller front item from left or right
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        # <= takes from the left half on ties, which keeps the sort stable
        if key(left[i]) <= key(right[j]):
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1

    # One half is used up; the rest of the other half is already sorted
    result.extend(left[i:])
    result.extend(right[j:])
    return result


def lower_bound(sorted_items, target, key=lambda x: x):
    """Binary search on a list already sorted by key.

    Return the FIRST index i where key(sorted_items[i]) >= target.
    If every item is smaller than target, return len(sorted_items).

    Example: [1, 3, 3, 5, 9] with target 3 -> 1,  target 4 -> 3,  target 10 -> 5
    Target: O(log n).  Keep two pointers lo, hi and halve the range each step.
    """
    # Search range is [lo, hi); hi starts at len because the answer can be past the end
    lo = 0
    hi = len(sorted_items)

    while lo < hi:
        mid = (lo + hi) // 2
        if key(sorted_items[mid]) < target:
            lo = mid + 1  # mid and everything before it are too small
        else:
            hi = mid  # mid could be the first match, so keep it in range

    # lo == hi: the first index whose key is >= target
    return lo


# =====================================================================
# PART 3 - Priority queue (binary min-heap)                    (Week 11)
# =====================================================================
class MinHeap:
    """A binary min-heap of (priority, item) pairs stored in a Python list.

    The SMALLEST priority comes out first. Priorities may be numbers or tuples.
    When two priorities are equal, the item pushed FIRST comes out first.
    Hint: store entries as (priority, counter, item) and compare (priority, counter).

    For index i: parent = (i - 1) // 2, children = 2*i + 1 and 2*i + 2.
    """

    def __init__(self):
        self._a = []  # the heap array
        self._count = 0  # insertion counter for tie-breaking

    def __len__(self):
        return len(self._a)

    def _less(self, i, j):
        """True if entry i should come out before entry j.
        Compares (priority, counter) only, so items themselves are never compared."""
        return (self._a[i][0], self._a[i][1]) < (self._a[j][0], self._a[j][1])

    def push(self, priority, item):
        """Add an item. Append at the end, then 'bubble up'. O(log n)."""
        self._a.append((priority, self._count, item))
        self._count += 1

        # Bubble up: swap with the parent while this entry is smaller
        i = len(self._a) - 1
        while i > 0:
            parent = (i - 1) // 2
            if self._less(i, parent):
                self._a[i], self._a[parent] = self._a[parent], self._a[i]
                i = parent
            else:
                break

    def peek(self):
        """Return (priority, item) with the smallest priority without removing it.
        Raise IndexError if the heap is empty. O(1)."""
        if len(self._a) == 0:
            raise IndexError("peek from an empty heap")
        priority, _, item = self._a[0]  # smallest entry is always at the top
        return (priority, item)

    def pop(self):
        """Remove and return (priority, item) with the smallest priority.
        Move the last entry to the top, then 'bubble down'.
        Raise IndexError if the heap is empty. O(log n)."""
        if len(self._a) == 0:
            raise IndexError("pop from an empty heap")

        top = self._a[0]
        last = self._a.pop()  # remove the last entry (cheap end of the list)

        # If anything is left, move the last entry to the top and bubble it down
        if len(self._a) > 0:
            self._a[0] = last
            i = 0
            n = len(self._a)
            while True:
                left = 2 * i + 1
                right = 2 * i + 2
                smallest = i
                # Find the smallest of this entry and its children
                if left < n and self._less(left, smallest):
                    smallest = left
                if right < n and self._less(right, smallest):
                    smallest = right
                if smallest == i:
                    break  # heap order is restored
                self._a[i], self._a[smallest] = self._a[smallest], self._a[i]
                i = smallest

        priority, _, item = top
        return (priority, item)


# =====================================================================
# PART 4 - Greedy                                               (Week 5)
# =====================================================================
def greedy_select(items, capacity):
    """Greedy choice: best value per unit of weight first.

    items    : list of dicts {"id": ..., "weight": positive int, "value": number}
    capacity : int, the most total weight you may take

    1) order the items by value / weight, highest first (use YOUR merge_sort;
       equal ratios keep their original order)
    2) walk that order and take every item that still fits

    Return (chosen_ids, total_value, total_weight), chosen_ids in the order taken.
    Target: O(n log n).
    """
    ordered = merge_sort(items, key=lambda it: -(it["value"] / it["weight"]))

    chosen = []
    total_value = 0
    total_weight = 0

    for it in ordered:
        if total_weight + it["weight"] <= capacity:
            chosen.append(it["id"])
            total_value += it["value"]
            total_weight += it["weight"]

    return (chosen, total_value, total_weight)


# =====================================================================
# PART 5 - Dynamic programming: 0/1 knapsack                (Weeks 6-7)
# =====================================================================
def dp_select(items, capacity):
    """Same input and output as greedy_select, but always the BEST total value.

    Table: best[i][c] = best value using only the first i items with capacity c
      best[0][c] = 0
      best[i][c] = best[i-1][c]                                  (skip item i)
                   or best[i-1][c - w] + v   if w <= c           (take item i)
    Then walk back from best[n][capacity] to find which items were taken.

    Return (chosen_ids, total_value, total_weight). chosen_ids in the same order
    as the items list. Target: O(n * capacity) time.
    """
    n = len(items)

    best = [[0] * (capacity + 1) for _ in range(n + 1)]
    for i in range (1, n + 1):
        w = items[i - 1]["weight"]
        v = items[i - 1]["value"]

        for c in range (capacity + 1):
            best[i][c] = best[i - 1][c]
            if w <= c and best[i - 1][c - w] + v > best[i][c]:
                best[i][c] = best[i - 1][c - w] + v


    taken = []
    c = capacity
    for i in range(n, 0, -1):
        if best[i][c] != best[i - 1][c]:
            taken.append(items[i - 1])
            c -= items[i - 1]["weight"]
    taken.reverse()

    chosen_ids = [it["id"] for it in taken]
    total_weight = sum(it["weight"] for it in taken)
    return (chosen_ids, best[n][capacity], total_weight)


# =====================================================================
# PART 6 - Graphs                                           (Weeks 9-10)
# graph format: dict  node -> list of (neighbor, weight), undirected
# =====================================================================
def bfs_reachable(graph, start, blocked_edge=None):
    """Breadth-first search from start.

    blocked_edge : None, or a pair (u, v) that may NOT be used in either
                   direction (for example a broken line or a closed road).

    Return a dict  node -> number of hops from start  for every reachable node
    (start itself has 0). Use collections.deque as the queue. Target: O(V + E).
    """
    hops = {start: 0}
    queue = deque([start])
    while queue:
        u = queue.popleft()
        for v, _ in graph.get(u, []):

            if blocked_edge == (u, v) or blocked_edge == (v, u):
                continue
            if v not in hops:
                hops[v] = hops[u] + 1
                queue.append(v)
    return hops


def dijkstra(graph, start):
    """Shortest paths from start (all weights >= 0). Use YOUR MinHeap.

    Return (dist, prev)
      dist[node] : length of the shortest path from start
      prev[node] : the node just before it on that path (prev[start] = None)
    Only reachable nodes appear in dist and prev. Target: O((V + E) log V).
    """
    dist = {start: 0}
    prev = {start: None}
    heap = MinHeap()
    heap.push(0, start)
    while len (heap) > 0:
        d, u = heap.pop()
        if d > dist[u]:
            continue
        for v, w in graph[u]:
            if v not in dist or d + w < dist[v]:
                dist[v] = d + w
                prev[v] = u
                heap.push(d + w, v)
    return dist, prev


def path_to(prev, target):
    """GIVEN (do not change). Rebuild the path start -> target from prev."""
    if target not in prev:
        return []
    path = []
    while target is not None:
        path.append(target)
        target = prev[target]
    path.reverse()
    return path


def mst_kruskal(nodes, edges):
    """Minimum spanning tree with Kruskal's algorithm + union-find.

    nodes : list of node ids
    edges : list of (u, v, cost)

    Sort the edges by cost (YOUR merge_sort), then add each edge whose ends are
    in different components. Return (tree_edges, total_cost), tree_edges as
    (u, v, cost) in the order added. Target: O(E log E).
    """
    parent = {n: n for n in nodes}

    tree = []
    total = 0
    for u, v, cost in merge_sort(edges, key=lambda e: e[2]):
        ru = u
        while parent[ru] != ru:
            ru = parent[ru]
        rv = v
        while parent[rv] != rv:
            rv = parent[rv]
        if ru != rv:
            parent[ru] = rv
            tree.append((u, v, cost))
            total += cost
    return tree, total




# =====================================================================
# PART 7 - String matching                                     (Week 12)
# =====================================================================
def kmp_search(text, pattern):
    """Knuth-Morris-Pratt. Return a list of EVERY start index where pattern
    occurs in text, overlapping matches included. Empty pattern -> [].

    Example: kmp_search("aaaa", "aa") -> [0, 1, 2]
    1) build the failure table: fail[i] = length of the longest proper prefix of
       pattern[:i+1] that is also a suffix of it
    2) scan the text once, never moving backwards in the text
    Target: O(n + m).  Do not use str.find, str.index, `in` on strings, or re.
    """
    raise NotImplementedError("Part 7: kmp_search")


# =====================================================================
# PART 8 - An NP-hard problem: the shortest tour (TSP)          (Week 14)
# dist format: dist[a][b] = travel cost from place a to place b
# =====================================================================
def tour_cost(order, dist):
    """GIVEN (do not change). Cost of order[0] -> ... -> order[-1] -> order[0]."""
    return sum(dist[order[i]][order[(i + 1) % len(order)]] for i in range(len(order)))


def tour_brute_force(start, stops, dist):
    """EXACT answer: try every order of the stops (itertools.permutations).

    Each tour begins at start, visits every stop once, and returns to start.
    Return (order, cost) where order is a list beginning with start
    (do not repeat start at the end). If stops is empty return ([start], 0).
    Target: O(n!) - only for about 9 stops or fewer.
    """
    raise NotImplementedError("Part 8: tour_brute_force")


def tour_heuristic(start, stops, dist):
    """FAST answer: nearest neighbor, then 2-opt improvement.

    1) nearest neighbor: from start, always go to the closest unvisited stop
    2) 2-opt: try reversing every segment order[i..j] (i >= 1, so start stays
       first); keep a reversal if it makes the tour shorter; repeat until no
       reversal helps
    Return (order, cost) in the same format as tour_brute_force.
    """
    raise NotImplementedError("Part 8: tour_heuristic")


# =====================================================================
# GIVEN - timing helper for the speed report               (Weeks 3, 13)
# =====================================================================
def time_it(func, *args, repeat=3):
    """GIVEN (do not change). Best wall-clock time in seconds over `repeat` runs."""
    best = None
    for _ in range(repeat):
        t0 = time.perf_counter()
        func(*args)
        t = time.perf_counter() - t0
        best = t if best is None or t < best else best
    return best
