import os
import cv2
import numpy as np
import pandas as pd
from scipy import stats
from skimage import measure
from skimage.morphology import convex_hull_image
from skimage.measure import label, regionprops
from skimage.feature import graycomatrix, graycoprops
from skimage.color import rgb2gray
from skimage.util import img_as_ubyte
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm

# Renommage des fichiers .png
def rename_filenames(filenames):
    new_filepaths = []
    for filepath in filenames:
        dirname = os.path.dirname(filepath)
        name = os.path.basename(filepath)
        # Split the filename at the first underscore
        parts = name.split('_')
        sample = parts[0]
        rest = '_'.join(parts[1:])
        
        # Construct the new name with "_EXP" before the first underscore
        new_name = f"{sample}-EXP_{rest}"

        new_filepath = os.path.join(dirname, new_name)
        os.rename(filepath, new_filepath)
        print(f"Rename : {filepath} -> {new_filepath}")
        new_filepaths.append(new_filepath)

    return new_filepaths

# Analyse d'image
class ImageProcessor:
    def __init__(self, image_path):
        self.image_path = image_path
        self.image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        self.mask, self.rgb_masked, self.contour, self.alpha_ch = self._mask()
        self.height, self.width = self.mask.shape

    def _mask(self):
        alpha_ch = self.image[..., 3]
        rgb_image = self.image[..., :3]
        ret, thr = cv2.threshold(alpha_ch, 120, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contour = max(contours, key=len)
        part_mask = np.zeros(rgb_image.shape)
        part_mask = cv2.drawContours(part_mask, [contour], -1,(255,255,255), -1)

        rgb_masked = rgb_image.copy()
        rgb_masked[np.logical_not(part_mask)] = 0

        return part_mask[..., 0], rgb_masked, contour, alpha_ch

class ShapeAnalyzer:
    def __init__(self, processor):
        self.processor = processor
        self.contour = processor.contour
        self.label_img = label(processor.mask)
        self.regions = regionprops(self.label_img)
        self.props = max(self.regions, key=lambda prop: prop.area)

    def eccentricity_from_moments(self, moments):
        mu20 = moments['mu20']
        mu02 = moments['mu02']
        mu11 = moments['mu11'] ** 2
        numerator = mu20 + mu02 + np.sqrt(4 * mu11 + (mu20 - mu02) ** 2)
        denominator = mu20 + mu02 - np.sqrt(4 * mu11 + (mu20 - mu02) ** 2)
        if denominator == 0:
            return 0
        eccentricity = np.sqrt(1 - (denominator / numerator))
        return eccentricity

    def eccentricity_from_ellipse(self, contour):
        if len(contour) < 5:
            return 0
        (x, y), (MA, ma), angle = cv2.fitEllipse(contour)
        a = max(MA, ma) / 2
        b = min(MA, ma) / 2
        eccentricity = np.sqrt(1 - (b ** 2) / (a ** 2))
        return eccentricity

    def aspect_ratio_from_ellipse(self, contour):
        if len(contour) < 5:
            return 1
        (x, y), (MA, ma), angle = cv2.fitEllipse(contour)
        aspect_ratio = max(MA, ma) / min(MA, ma) if min(MA, ma) > 0 else 1
        return aspect_ratio

    def analyze(self):
        hull = convex_hull_image(self.processor.mask)
        hull_perim = measure.perimeter(hull)
        moments = cv2.moments(self.contour)
        
        circ_dellino = self.props.perimeter / (2 * np.sqrt(np.pi * self.props.area))
        rectangularity = self.props.perimeter / (2 * (self.props.major_axis_length + self.props.minor_axis_length))
        compactness = self.props.area / (self.props.major_axis_length * self.props.minor_axis_length)
        elongation = (self.props.feret_diameter_max ** 2) / self.props.area

        circ_rect = circ_dellino * rectangularity
        comp_elon = compactness * elongation
        circ_elon = circ_dellino * elongation
        rect_comp = rectangularity * compactness

        shape_features = {
            'aspect_rat': self.aspect_ratio_from_ellipse(self.contour),
            'solidity': self.props.solidity,
            'convexity': hull_perim / self.props.perimeter,
            'circularity_cioni': (4 * np.pi * self.props.area) / (self.props.perimeter ** 2),
            'circularity_dellino': circ_dellino,
            'rectangularity': rectangularity,
            'compactness': compactness,
            'elongation': elongation,
            'roundness': 4 * self.props.area / (np.pi * (self.props.feret_diameter_max ** 2)),
            'eccentricity_moments': self.eccentricity_from_moments(moments),
            'eccentricity_ellipse': self.eccentricity_from_ellipse(self.contour),
            'circ_rect': circ_rect, 
            'comp_elon': comp_elon, 
            'circ_elon': circ_elon, 
            'rect_comp': rect_comp
        }

        return shape_features

class TextureAnalyzer:
    def __init__(self, processor, props):
        self.processor = processor
        self.props = props
        self.gray_image = rgb2gray(processor.rgb_masked)
        self.ubyte_image = img_as_ubyte(self.gray_image)
        self.patch_size = max(int(self.props.major_axis_length / 10), 1)
        self.step = max(int(self.props.major_axis_length / 20), 1)
        self.thetas = [
            0, np.pi/8, np.pi/4, np.pi/2, 3*np.pi/4, np.pi,
            np.pi+np.pi/4, np.pi+np.pi/2, np.pi+3*np.pi/4, 
            np.pi/4 + np.pi/8, np.pi/2 + np.pi/8, 3*np.pi/4 + np.pi/8, 
            np.pi + np.pi/8, np.pi+np.pi/4 + np.pi/8, np.pi+np.pi/2 + np.pi/8, 
            np.pi+3*np.pi/4 + np.pi/8
        ]
        self.bins = np.array([0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 255])  # 16-bit
        self.inds = np.digitize(self.ubyte_image, self.bins)
        self.max_value = self.inds.max() + 1

    def _extract_patches(self):
        y_centroid, x_centroid = self.props.centroid
        height, width = self.processor.mask.shape
        locs = [(int(y_centroid), int(x_centroid))]

        for theta in self.thetas:
            for i in range(1, 1001):
                new_y = int(y_centroid - np.sin(theta) * self.step * i)
                new_x = int(x_centroid + np.cos(theta) * self.step * i)
                if 0 <= new_x < width and 0 <= new_y < height and self.processor.mask[new_y, new_x] > 0:
                    locs.append((new_y, new_x))
                else:
                    break

        locs_no_background = []
        patches = []

        for loc in locs:
            patch = self.processor.mask[loc[0]:loc[0] + self.patch_size, loc[1]:loc[1] + self.patch_size]
            if patch.min() > 0:
                glcm_patch = self.inds[loc[0]:loc[0] + self.patch_size, loc[1]:loc[1] + self.patch_size]
                locs_no_background.append(loc)
                patches.append(glcm_patch)

        return patches

    def analyze(self):
        patches = self._extract_patches()
        distances = [1] + [int((i + 1) / 5 * self.patch_size) for i in range(5)]
        features = {prop: [] for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'asm']}
        
        for patch in patches:
            glcm = graycomatrix(patch, distances, self.thetas, levels=self.max_value, normed=False, symmetric=False)
            for prop in features.keys():
                if prop == "asm":
                    features[prop].append(graycoprops(glcm, prop.upper()).mean())
                else:
                    features[prop].append(graycoprops(glcm, prop).mean())

        avg_features = {f'{k}': np.mean(v) for k, v in features.items()}

        return avg_features


class ColorAnalyzer:
    def __init__(self, processor):
        self.processor = processor

    def analyze(self):
        rgb_image = self.processor.rgb_masked
        hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
        color_features = {}
        for i, color in enumerate(['red', 'green', 'blue']):
            channel = rgb_image[..., i].flatten()
            channel_pixels = channel[channel > 0]
            color_features[f'{color}_mean'] = np.mean(channel_pixels)
            color_features[f'{color}_std'] = np.std(channel_pixels)
            color_features[f'{color}_mode'] = stats.mode(channel_pixels, axis=None)[0]
        
        for i, color in enumerate(['hue', 'saturation', 'value']):
            channel = hsv_image[..., i].flatten()
            channel_pixels = channel[channel > 0]
            color_features[f'{color}_mean'] = np.mean(channel_pixels)
            color_features[f'{color}_std'] = np.std(channel_pixels)
            color_features[f'{color}_mode'] = stats.mode(channel_pixels, axis=None)[0]
        
        return color_features

def parse_extra_attributes(rest_label):
    
    # Mapping pour color, crystallinity, shape et hydro_alter_degree
    extra_labels = {
        "color": {"tr": "transparent", "bl": "black"},
        "crystallinity": {"lc": "low crystallinity", "mc": "mid crystallinity", "hc": "high crystallinity"},
        "shape": {"b": "blocky", "f": "fluidal", "mt": "microtubular", "hv": "highly vesicular", "s": "spongy", "p": "pumice", "agg": "aggregate"},
        "hydro_alter_degree": {"n": "none", "l": "low", "m": "medium", "h": "high"}
    }

    color, crystallinity, shape, hydro_alter_degree = None, None, None, None
    for key, mapping in extra_labels.items():
        for sub_key, sub_value in mapping.items():
            if rest_label.startswith(sub_key):
                rest_label = rest_label[len(sub_key):]
                if key == "color":
                    color = sub_value
                elif key == "crystallinity":
                    crystallinity = sub_value
                elif key == "shape":
                    shape = sub_value
                elif key == "hydro_alter_degree":
                    hydro_alter_degree = sub_value
                break
    return color, crystallinity, shape, hydro_alter_degree

def add_extra_features(csv_path):
    # Charger le fichier CSV
    df = pd.read_csv(csv_path)

    filenames = df.iloc[:, 0].to_numpy()

    # Mapping des labels aux types et sous-types
    label_mapping = {
        "free crystal": {
            "PG": "plagioclase",
            "PX": "pyroxene",
            "AMF": "amphibole",
            "SU": "S-group mineral",
            "OL": "olivine",
            "OT": "others"
        },
        "altered material": {
            "AW": "weathered material",
            "AH": "hydrothermally altered material"
        },
        "lithic": {
            "LL": "standard lithic",
            "LRJ": "recycled juvenile particles"
        },
        "juvenile": {
            "JJ": "standard juvenile",
            "JC": "coated juvenile particles"
        }
    }

    afe_code_mapping = {
        "A17": "MAR_47_DB1",
        "B1": "MAR_47_DB2",
        "C5": "MAR_47_DB3",
        "E3": "MAR_47_DB4",
        "G1": "MAR_47_DB5",
        "H1": "MAR_47_DB6",
        "K1": "MAR_47_DB7",
        "M1": "MAR_47_DB8",
        "O1": "MAR_47_DB9",
        "CV-DB1": "LAP_2_DB1",
        "KE-DB2": "KEL_16_DB2",
        "KE-DB3": "KEL_16_DB3",
        "ME-DB1": "MEA_22post_DB1",
        "ME-DB2": "MEA_22post_DB2",
        "SG-DB1": "SOG_6_DB1",
        "SG-DB2": "SOG_6_DB2",
        "NC-DB2": "CIN_11_DB2",
        "NC-DB15": "CIN_11_DB15",
        "ON-DB1": "ONT_10_DB1",
        "PI-DB1": "PIN_3_DB1",
        "MS-DB1": "STH_12post_DB1",
        "TO-DB1": "TOB_6_DB1"
    }

    volc_mapping = {
        "VE-00-EXP": (211020, "Vesuvius"),
        "VE-01-EXP": (211020, "Vesuvius"),
        "VE-02-EXP": (211020, "Vesuvius"),
        "VE-03-EXP": (211020, "Vesuvius"),
        "VE-04-EXP": (211020, "Vesuvius"),
        "VE-05-EXP": (211020, "Vesuvius"),
        "VE-08-EXP": (211020, "Vesuvius"),
        "VE-10-EXP": (211020, "Vesuvius"),
        "VE-11-EXP": (211020, "Vesuvius"),
        "ET-00-EXP": (211060, "Etna"),
        "ET-02-EXP": (211060, "Etna"),
        "ET-03-EXP": (211060, "Etna"),
        "ET-04-EXP": (211060, "Etna"),
        "ET-05-EXP": (211060, "Etna"),
        "ET-06-EXP": (211060, "Etna"),
        "ET-07-EXP": (211060, "Etna"),
        "ET-08-EXP": (211060, "Etna"),
        "ET-11-EXP": (211060, "Etna"),
        "ET-15-EXP": (211060, "Etna"),
        "ET-16-EXP": (211060, "Etna"),
        "ET-20-EXP": (211060, "Etna"),
        "ET-29-EXP": (211060, "Etna"),
        "ET-30-EXP": (211060, "Etna"),
        "ST-00-EXP": (211040, "Stromboli"),
        "ST-01-EXP": (211040, "Stromboli"),
        "ST-02-EXP": (211040, "Stromboli"),
        "ST-03-EXP": (211040, "Stromboli"),
        "ST-07-EXP": (211040, "Stromboli"),
        "ST-09-EXP": (211040, "Stromboli"),
        "ST-11-EXP": (211040, "Stromboli"),
        "LAP_2_DB1": (383010, "La Palma"),
        "KEL_16_DB2": (263280, "Kelut"),
        "KEL_16_DB3": (263280, "Kelut"),
        "MEA_22post_DB1": (263250, "Merapi"),
        "MEA_22post_DB2": (263250, "Merapi"),
        "SOG_6_DB1": (360060, "Soufrière Guadeloupe"),
        "SOG_6_DB2": (360060, "Soufrière Guadeloupe"),
        "CIN_11_DB2": (357070, "Chillán, Nevados de"),
        "CIN_11_DB15": (357070, "Chillán, Nevados de"),
        "ONT_10_DB1": (283040, "On-take"),
        "PIN_3_DB1": (273083, "Pinatubo"),
        "STH_12post_DB1": (321050, "St. Helens"),
        "TOB_6_DB1": (261090, "Toba"),
        "MAR_47_DB1": (261140, "Marapi"),
        "MAR_47_DB2": (261140, "Marapi"),
        "MAR_47_DB3": (261140, "Marapi"),
        "MAR_47_DB4": (261140, "Marapi"),
        "MAR_47_DB5": (261140, "Marapi"),
        "MAR_47_DB6": (261140, "Marapi"),
        "MAR_47_DB7": (261140, "Marapi"),
        "MAR_47_DB8": (261140, "Marapi"),
        "MAR_47_DB9": (261140, "Marapi")
    }

    # Initialiser des listes pour les nouvelles colonnes
    afe_codes, gsLows, gsUps, ids, imgURLs, instruments, magnifications, multifocuss, main_types, sub_types, colors, crystallinities, shapes, hydro_alter_degrees, types, lusters, edges, weathering_signs, volc_names, volc_nums = ([] for _ in range(20))


    for filename in filenames:
        parts = filename.split("_")
        sample = parts[0]

        if sample in afe_code_mapping:
            sample = afe_code_mapping.get(sample)

        volc_num, volc_name = volc_mapping.get(sample)
        
        if sample.endswith("EXP"):
            aliquote, scan, particle_idx, microscope_type, magnification, grain_size = parts[1:]
            imgURL = f"{filename}.png"
            multifocus = microscope_type == 'mf'

            if grain_size == "unsieved":
                gsLow = "NA"
                gsUp = "NA"
            else:
                gsLow = gsUp = None

            afe_codes.append(sample)
            gsLows.append(gsLow)
            gsUps.append(gsUp)
            ids.append(particle_idx)
            imgURLs.append(imgURL)
            instruments.append(microscope_type)
            magnifications.append(magnification.rstrip('x'))
            multifocuss.append(multifocus)
            main_types.append(None)
            sub_types.append(None)
            colors.append(None)
            crystallinities.append(None)
            shapes.append(None)
            hydro_alter_degrees.append(None)
            types.append("experimental")
            edges.append(None)
            lusters.append(None)
            weathering_signs.append(None)
            volc_names.append(volc_name)
            volc_nums.append(volc_num)

        else:
            aliquote, scan, particle_idx, microscope_type, magnification, grain_size, label = parts[1:]
            imgURL = f"{filename}.png"
            multifocus = microscope_type == 'mf'

            if grain_size == "mesh60":
                gsLow = 2
                gsUp = 1
            elif grain_size == "mesh120":
                gsLow = 3
                gsUp = 2
            elif grain_size == "phi0phi1":
                gsLow = 0
                gsUp = 1
            elif grain_size == "phi1phi2":
                gsLow = 1
                gsUp = 2
            elif grain_size == "morephi0":
                gsLow = 0
                gsUp = -1
            else:
                gsLow = "NA"
                gsUp = "NA"

            main_type, sub_type = None, None
            for key, value in label_mapping.items():
                for (subtype_key, subtype_value) in value.items():
                    if label.startswith(subtype_key):
                        main_type = key
                        sub_type = subtype_value
                        rest_label = label[len(subtype_key):]
                        break

            if main_type is None or sub_type is None:
                print(f"Label '{label}' non reconnu")
                continue

            color, crystallinity, shape, hydro_alter_degree = parse_extra_attributes(rest_label)

            afe_codes.append(sample)
            gsLows.append(gsLow)
            gsUps.append(gsUp)
            ids.append(particle_idx)
            imgURLs.append(imgURL)
            instruments.append(microscope_type)
            magnifications.append(magnification.rstrip('x'))
            multifocuss.append(multifocus)
            main_types.append(main_type)
            sub_types.append(sub_type)
            colors.append(color)
            crystallinities.append(crystallinity)
            shapes.append(shape)
            hydro_alter_degrees.append(hydro_alter_degree)
            types.append("natural")
            edges.append(None)
            lusters.append(None)
            weathering_signs.append(None)
            volc_names.append(volc_name)
            volc_nums.append(volc_num)


    # Ajouter les nouvelles colonnes au DataFrame
    df['afe_code'] = afe_codes
    df['gsLow'] = gsLows
    df['gsUp'] = gsUps
    df['id'] = ids
    df['imgURL'] = imgURLs
    df['instrument'] = instruments
    df['magnification'] = magnifications
    df['multi_focus'] = multifocuss
    df['main_type'] = main_types
    df['sub_type'] = sub_types
    df['color'] = colors
    df['crystallinity'] = crystallinities
    df['shape'] = shapes
    df['hydro_alter_degree'] = hydro_alter_degrees
    df['type'] = types
    df['edge'] = edges
    df['luster'] = lusters
    df['weathering_sign'] = weathering_signs
    df['volc_name'] = volc_names
    df['volc_num'] = volc_nums
    
    # Sauvegarder le DataFrame modifié dans un nouveau fichier CSV
    new_csv_path = csv_path.replace('.csv', '_modified.csv')
    df.rename(columns={'Unnamed: 0': 'filename'}, inplace=True)
    df.to_csv(new_csv_path, index=False)



def main(filenames, csv_filename):
    qia_dict = {}
    total = 0

    for idx, filepath in tqdm(enumerate(filenames), total=len(filenames), desc="Processing images"):
        
        processor = ImageProcessor(filepath)
        if processor.image is None:
            print("Skip image: {filepath}")
            continue

        basename = os.path.basename(filepath)
        name, _ = basename.split(".")

        analyzer_shape = ShapeAnalyzer(processor)
        analyzer_texture = TextureAnalyzer(processor, analyzer_shape.props)
        analyzer_color = ColorAnalyzer(processor)

        total += 1
        dict1 = analyzer_shape.analyze()
        dict2 = analyzer_texture.analyze()
        dict3 = analyzer_color.analyze()

        qia_dict[name] = {**dict1, **dict2, **dict3}


    df = pd.DataFrame.from_dict(qia_dict, orient='index')
    csv_path = f'./results/{csv_filename}.csv'
    df.to_csv(csv_path)
    add_extra_features(csv_path)
    print(f"Total : {total}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image processing for feature extraction")
    parser.add_argument('directory', type=str, help="Path to directory containing image files")
    parser.add_argument('csv_filename', type=str, help="Name of the csv file")
    parser.add_argument('--rename_experimental_data', action='store_true', help="If you want to rename filenames of experimental data")
    args = parser.parse_args()
    
    filenames = [os.path.join(args.directory, f) for f in os.listdir(args.directory) if f.endswith('.png') and not f.startswith("._")]
    if args.rename_experimental_data:
        filenames = rename_filenames(filenames)
        
    main(filenames, args.csv_filename)
