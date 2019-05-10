import numpy as np


def read_obj(filepath):
    vertices = []
    faces = []
    material = None
    for line in open(filepath, "r"):
        if line.startswith('#'): continue
        values = line.split()
        if not values: continue
        if values[0] == 'v':
            # v = map(float, values[1:4])
            v = [float(x) for x in values[1:4]]
            vertices.append(v)

        elif values[0] == 'f':
            face = []
            for v in values[1:]:
                w = v.split('/')
                face.append(int(w[0]))
            faces.append(face)
    vertices = np.array(vertices)
    return vertices, faces


def write_obj(filepath, vertices, faces):
    with open(filepath, 'w') as fp:
      for v in vertices:
        fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))

      for f in faces:
        fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))