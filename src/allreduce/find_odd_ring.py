def find_ring(n):
    # find a ring excluding centre in an n*n mesh where n is odd

    # centre = (n-1)/2, (n-1)/2
    
    ring = []
    y = (n-1)/2
    x = (n-1)/2 - 1
    ring.append(int(n*y + x))
    y = y - 1
    ring.append(int(n*y + x))
    x = x + 1
    ring.append(int(n*y + x))
    x = x + 1
    ring.append(int(n*y + x))
    y = y + 1
    ring.append(int(n*y + x))
    y = y + 1
    ring.append(int(n*y + x))
    x = x - 1
    ring.append(int(n*y + x))
    x = x - 1
    ring.append(int(n*y + x))

    while x > 0:
        x = x - 1
        const = x
        # form : up right down left up circle
        if y < (n-1)/2:
            while y != const:
                ring.append(int(n*y + x))
                y = y - 1
            while x != (n-1-const):
                ring.append(int(n*y + x))
                x = x + 1
            while y != (n-1-const):
                ring.append(int(n*y + x))
                y = y + 1
            while x != const:
                ring.append(int(n*y + x))
                x = x - 1
            while y != ((n-1)/2):
                ring.append(int(n*y + x))
                y = y - 1
            y = y + 1
        # form : down right up left down circle
        else:
            while y != (n-1-const):
                ring.append(int(n*y + x))
                y = y + 1
            while x != (n-1-const):
                ring.append(int(n*y + x))
                x = x + 1
            while y != const:
                ring.append(int(n*y + x))
                y = y - 1
            while x != const:
                ring.append(int(n*y + x))
                x = x - 1
            while y != ((n-1)/2):
                ring.append(int(n*y + x))
                y = y + 1
            y = y - 1
    y = (n-1)/2
    while x != (n-1)/2 - 1 and len(ring) != n*n - 1:
        ring.append(int(n*y + x))
        x = x + 1
    return ring

def find_ngbrs(n):
    y = (n-1)/2
    x = (n-1)/2
    center_ngbrs = []
    center_ngbrs.append(int(n*y + x - 1))
    center_ngbrs.append(int(n*y + x + 1))
    center_ngbrs.append(int(n*(y-1) + x))
    center_ngbrs.append(int(n*(y+1) + x))

    return center_ngbrs
