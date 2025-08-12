import bz2

input_file = "shape_predictor_68_face_landmarks.dat.bz2"
output_file = "shape_predictor_68_face_landmarks.dat"

with bz2.BZ2File(input_file, "rb") as file:
    data = file.read()

with open(output_file, "wb") as out_file:
    out_file.write(data)

print(f"Extracted {output_file} from {input_file}")
