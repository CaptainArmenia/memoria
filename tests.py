from inferencia import *

def test_divide_set():
    x = range(0, 10)
    result = divide_set(x, 0.8)

    assert len(result[0]) == 8
    assert len(result[1]) == 2

def test_more_frequent():
    x = ['a', 'a', 'b', 'c', 'c', 'c']

    assert most_frequent(x) == 'c'

def test_cos_distance():
    origen = np.array([0, 1])
    p0 = np.array([0, 2])
    p1 = np.array([1, 0])
    p2 = np.array([0, -1])
    p3 = np.array([-1, 0])
    p4 = np.array([0, 3])

    assert cos_distance(origen, p0) == 0
    assert cos_distance(origen, p1) == 1
    assert cos_distance(origen, p2) == 2
    assert cos_distance(origen, p3) == 1
    assert cos_distance(origen, p0) == cos_distance(origen, p4)

def test_knn():
    puntos = [np.array([x, 1]) for x in range(1, 6)]
    origen = np.array([0, 1])
    p0 = np.array([0, 2])
    p1 = np.array([1, 0])

    assert [x[0] for x in knn(origen, puntos, 3)] == [0, 1, 2]

test_divide_set()
test_more_frequent()
test_cos_distance()
test_knn()

print("tests superados")