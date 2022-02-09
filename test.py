import collections

x = collections.deque([], 10)
x.append({
    "value": "A",
    "frame": 3
})
x.append({
    "value": "B",
    "frame": 3
})

for i in range(5):
    x.append({
        "value": "C",
        "frame": 10
    })
    x.append({
        "value": "D",
        "frame": 10
    })
    pop_times = 0
    for pos in range(len(x)):
        x[pos]["frame"] -= 1
        if x[pos]["frame"] <= 0:
            pop_times += 1
    for _ in range(pop_times):
        x.popleft()
    print(list(x))