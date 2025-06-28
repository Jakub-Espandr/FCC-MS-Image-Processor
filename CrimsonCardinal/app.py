import logging
from flask import Flask, request, render_template, jsonify, send_from_directory, url_for
import os
import imageio.v2 as imageio
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import cm

from PIL import Image

plt.switch_backend('Agg')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULT_FOLDER'] = 'static/results'
app.secret_key = "supersecretkey"

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

# Setup basic logging
logging.basicConfig(level=logging.DEBUG)

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'templates'), 'favicon.ico', mimetype='image/vnd.microsoft.icon')

# === Index calculation functions (unchanged) ===
def calculate_ndvi(NIR, R):
    return np.true_divide(np.subtract(NIR, R), np.add(NIR, R))

def calculate_ndvi_ngb(NIR, B):
    return np.true_divide(np.subtract(NIR, B), np.add(NIR, B))

def calculate_endvi(NIR, G, B):
    NIRG = np.add(NIR, G)
    return np.true_divide(np.subtract(NIRG, 2*B), np.add(NIRG, 2*B))

def calculate_endvi_rgn(NIR, R, G):
    NIRR = np.add(NIR, R)
    return np.true_divide(np.subtract(NIRR, 2*G), np.add(NIRR, 2*G))

def calculate_savi(NIR, R, L=0.5):
    NIRminusR = np.subtract(NIR, R)
    NIRplusRplusL = np.add(np.add(NIR, R), L)
    return np.multiply(np.true_divide(NIRminusR, NIRplusRplusL), (1 + L))

def calculate_tvi(NIR, R, G):
    return 0.5 * (120 * (NIR - G) - 200 * (R - G))

def calculate_gndvi(NIR, G):
    return np.true_divide(np.subtract(NIR, G), np.add(NIR, G))

def calculate_msavi(NIR, R):
    return 0.5 * (2 * NIR + 1 - np.sqrt((2 * NIR + 1) ** 2 - 8 * (NIR - R)))

def calculate_cvi(NIR, R):
    return np.true_divide(NIR, R) - 1

def calculate_cvi2(NIR, R, G):
    return np.true_divide(np.multiply(NIR, R), np.square(G))

def calculate_pri(NIR, R):
    return np.true_divide(np.subtract(NIR, R), np.add(NIR, R))

def calculate_ndwi(G, NIR):
    return np.true_divide(np.subtract(G, NIR), np.add(G, NIR))

def calculate_vari(G, R, B):
    return np.true_divide(np.subtract(G, R), np.add(np.subtract(G, R), B))

def calculate_evi(NIR, R, B):
    return 2.5 * np.true_divide(np.subtract(NIR, R), np.add(np.add(NIR, 6 * R), np.add(-7.5 * B, 1)))

def calculate_ng(G, NIR, R):
    return np.true_divide(G, np.add(np.add(NIR, R), G))

# === New RGB indices ===
def calculate_ngrdi(G, R):
    return np.true_divide(np.subtract(G, R), np.add(G, R))

def calculate_exg(G, R, B):
    return 2 * G - R - B

def calculate_exr(R, G):
    return 1.4 * R - G

def calculate_exgr(G, R, B):
    return (2 * G - R - B) - (1.4 * R - G)

def calculate_gli(G, R, B):
    numerator = 2 * G - R - B
    denominator = 2 * G + R + B
    return np.true_divide(numerator, denominator)

def calculate_gli2(G, R, B):
    numerator = (G - R) + (G - B)
    denominator = 2 * G + R + B
    return np.true_divide(numerator, denominator)

def calculate_rgbvi(G, R, B):
    return np.true_divide(np.square(G) - np.multiply(R, B), np.square(G) + np.multiply(R, B))

def calculate_tgi(R, G, B):
    return -0.5 * ((660 - 550) * (R - G) - (660 - 480) * (R - B))

def calculate_ngbdi(G, B):
    return np.true_divide(np.subtract(G, B), np.add(G, B))

def calculate_vdvi(G, R, B):
    numerator = 2 * G - R - B
    denominator = 2 * G + R + B
    return np.true_divide(numerator, denominator)

def calculate_mexg(G, R, B):
    return 1.262 * G - 0.884 * R - 0.311 * B

def calculate_veg(G, R, B):
    return np.true_divide(G, (np.power(R, 0.667) * np.power(B, 0.333)))

def calculate_gcc(G, R, B):
    return np.true_divide(G, np.add(np.add(R, G), B))

def calculate_cive(R, G, B):
    return 0.441 * R - 0.811 * G + 0.385 * B + 18.78745

def calculate_ndti(R, G):
    return np.true_divide(np.subtract(R, G), np.add(R, G))

def calculate_sci(R, G):
    return np.true_divide(np.subtract(R, G), np.add(R, G))

def calculate_ndbi(R, G):
    return np.true_divide(np.subtract(R, G), np.add(R, G))

def calculate_bi(R, G, B):
    return np.sqrt((np.square(R) + np.square(G) + np.square(B)) / 3)

def calculate_ui(R, B):
    return np.true_divide(R, np.add(np.add(R, B), R + B))  # Safe option

# === New NGB indices ===
def calculate_ndvi_mod(NIR, G):
    return np.true_divide(np.subtract(NIR, G), np.add(NIR, G))

def calculate_gndvi(NIR, G):
    return np.true_divide(np.subtract(NIR, G), np.add(NIR, G))

def calculate_ndbi_blue(NIR, B):
    return np.true_divide(np.subtract(NIR, B), np.add(NIR, B))

def calculate_ndgi(G, B):
    return np.true_divide(np.subtract(G, B), np.add(G, B))

def calculate_bgi(G, B):
    return np.true_divide(np.subtract(G, B), np.add(G, B))

def calculate_evi_mod(NIR, G, B):
    return 2.5 * np.true_divide(np.subtract(NIR, G), np.add(np.add(NIR, 6 * G), np.add(-7.5 * B, 1)))

def calculate_msavi(NIR, G):
    return 0.5 * (2 * NIR + 1 - np.sqrt((2 * NIR + 1) ** 2 - 8 * (NIR - G)))

def calculate_endvi_ngb(NIR, G, B):
    return np.true_divide(np.subtract(np.add(NIR, G), 2 * B), np.add(np.add(NIR, G), 2 * B))

def calculate_gndwi(G, NIR):
    return np.true_divide(np.subtract(G, NIR), np.add(G, NIR))

def calculate_cig(NIR, G):
    return np.true_divide(NIR, G) - 1

def calculate_gbndvi(NIR, G, B):
    return np.true_divide(np.subtract(NIR, np.add(G, B)), np.add(NIR, np.add(G, B)))

def calculate_gsavi(NIR, G, L=0.16):
    numerator = np.subtract(NIR, np.add(G, L))
    denominator = np.add(np.add(NIR, G), L)
    return np.true_divide(numerator, denominator)

def calculate_grndvi(NIR, G):
    return np.true_divide(np.subtract(NIR, G), np.add(NIR, G))

def calculate_gosavi(NIR, G):
    return np.true_divide(np.subtract(NIR, G), np.add(np.add(NIR, G), 0.16))

def calculate_ndwi(G, NIR):
    return np.true_divide(np.subtract(G, NIR), np.add(G, NIR))

def calculate_bndvi(NIR, B):
    return np.true_divide(np.subtract(NIR, B), np.add(NIR, B))

def calculate_ngbvi(G, B):
    return np.true_divide(np.subtract(G, B), np.add(G, B))

def calculate_cig_simple(NIR, G):
    return np.true_divide(NIR, G) - 1

def calculate_bwdrvi(NIR, B):
    return np.true_divide(0.1 * NIR - B, 0.1 * NIR + B)

# === New RGN indices ===
def calculate_ndvi(NIR, R):
    return np.true_divide(np.subtract(NIR, R), np.add(NIR, R))

def calculate_gndvi(NIR, G):
    return np.true_divide(np.subtract(NIR, G), np.add(NIR, G))

def calculate_savi(NIR, R, L=0.5):
    NIRminusR = np.subtract(NIR, R)
    NIRplusRplusL = np.add(np.add(NIR, R), L)
    return np.multiply(np.true_divide(NIRminusR, NIRplusRplusL), (1 + L))

def calculate_msavi(NIR, R):
    return 0.5 * (2 * NIR + 1 - np.sqrt((2 * NIR + 1) ** 2 - 8 * (NIR - R)))

def calculate_osavi(NIR, R):
    return np.multiply(np.true_divide(np.subtract(NIR, R), np.add(NIR, R) + 0.16), (1 + 0.16))

def calculate_evi2(NIR, R):
    return 2.5 * np.true_divide(np.subtract(NIR, R), np.add(NIR, 2.4 * R + 1))

def calculate_sr(NIR, R):
    return np.true_divide(NIR, R)

def calculate_rdvi(NIR, R):
    return np.true_divide(np.subtract(NIR, R), np.sqrt(np.add(NIR, R)))

def calculate_wdrvi(NIR, R, a=0.1):
    return np.true_divide(a * NIR - R, a * NIR + R)

def calculate_mtvi2(NIR, R, G):
    numerator = 1.5 * (1.2 * (NIR - G) - 2.5 * (R - G))
    denominator = np.sqrt((NIR + 1) ** 2 - (R + 1) ** 2)
    return np.true_divide(numerator, denominator)

def calculate_dvi(NIR, R):
    return np.subtract(NIR, R)

def calculate_ndwi(G, NIR):
    return np.true_divide(np.subtract(G, NIR), np.add(G, NIR))

def calculate_cig(NIR, G):
    return np.true_divide(NIR, G) - 1

def calculate_cired(NIR, R):
    return np.true_divide(NIR, R) - 1

def calculate_cvi(NIR, R, G):
    return np.true_divide(np.multiply(NIR, R), np.square(G))

def calculate_grvi(G, R):
    return np.true_divide(np.subtract(G, R), np.add(G, R))

def calculate_rgri(R, G):
    return np.true_divide(R, G)

def calculate_ngrdi(G, R):
    return np.true_divide(np.subtract(G, R), np.add(G, R))

# === Define Haxby colormap ===
haxby_colors = [
    (0, '#000080'), (0.1, '#0000ff'), (0.2, '#00ffff'), (0.3, '#00ff00'), 
    (0.4, '#ffff00'), (0.5, '#ffa500'), (0.6, '#ff4500'), (0.7, '#ff0000'), 
    (0.8, '#8b0000'), (1.0, '#000000')
]
haxby_cmap = LinearSegmentedColormap.from_list('Haxby', haxby_colors)

# === Colormap options ===
colormap_options = {
    'Viridis': 'viridis',
    'Jet': 'jet',
    'Magma': 'magma',
    'Plasma': 'plasma',
    'Inferno': 'inferno',
    'Cividis': 'cividis',
    'RdYlGn': 'RdYlGn',
    'Blues': 'Blues',
    'Greys': 'Greys',
    'Haxby': haxby_cmap,
    'Cool': 'cool',
    'Spring': 'spring',
    'Summer': 'summer',
    'Autumn': 'autumn',
    'Winter': 'winter',
    'Hot': 'hot',
    'Coolwarm': 'coolwarm'
}

# === Process function ===
def process_and_save(image_path, output_name, calculation_func, label, colormap_name, *args):
    try:
        rgb = imageio.imread(image_path)
        result = calculation_func(*args)
        abs_max = max(abs(np.min(result)), abs(np.max(result)))

        fig, ax = plt.subplots()
        cmap = colormap_options[colormap_name]
        ax.imshow(result, cmap=cmap)
        plt.axis('off')

        from io import BytesIO
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=600, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        buf.seek(0)

        main_image = imageio.imread(buf)
        if main_image.shape[2] == 4:
            main_image = main_image[:, :, :3]

        fig, ax = plt.subplots(figsize=(main_image.shape[1] * 0.8 / 100, 1))
        norm = plt.Normalize(vmin=-abs_max, vmax=abs_max)
        cbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=ax, orientation='horizontal')
        cbar.ax.tick_params(labelsize=10)
        cbar.update_ticks()
        cbar.set_label(label, fontsize=16, fontweight='bold', labelpad=-50)

        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=600, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        buf.seek(0)

        colorbar_image = imageio.imread(buf)
        if (colorbar_image.shape[2] == 4):
            colorbar_image = colorbar_image[:, :, :3]

        colorbar_image_pil = Image.fromarray(colorbar_image)
        colorbar_image_pil = colorbar_image_pil.resize((int(main_image.shape[1] * 0.8), int(main_image.shape[0] * 0.1)))
        colorbar_image = np.array(colorbar_image_pil)

        padding_height = colorbar_image.shape[0]
        white_padding = np.full((padding_height, main_image.shape[1], 3), [255, 255, 255], dtype=np.uint8)

        x_offset = (white_padding.shape[1] - colorbar_image.shape[1]) // 2
        y_offset = (white_padding.shape[0] - colorbar_image.shape[0]) // 2
        white_padding[y_offset:y_offset + colorbar_image.shape[0], x_offset:x_offset + colorbar_image.shape[1]] = colorbar_image

        space_height = int(main_image.shape[0] * 0.01)
        space_padding = np.full((space_height, main_image.shape[1], 3), [255, 255, 255], dtype=np.uint8)

        result_with_padding = np.vstack((main_image, space_padding, white_padding))

        imageio.imwrite(output_name, result_with_padding)
        logging.info(f"Image processed and saved to {output_name}")

    except Exception as e:
        logging.error(f"Error in processing and saving image: {e}")
        raise

index_functions = {
    '1': ('NDVI', calculate_ndvi, 'RdYlGn', ['RGN']),
    '2': ('NDVI_NGB', calculate_ndvi_ngb, 'RdYlGn', ['NGB']),
    '3': ('ENDVI', calculate_endvi, 'Viridis', ['NGB']),
    '4': ('ENDVI_RGN', calculate_endvi_rgn, 'Viridis', ['RGN']),
    '5': ('SAVI', calculate_savi, 'Magma', ['RGN']),
    '6': ('TVI', calculate_tvi, 'Viridis', ['RGN']),
    '7': ('GNDVI', calculate_gndvi, 'RdYlGn', ['RGN', 'NGB', 'RGB']),
    '8': ('MSAVI', calculate_msavi, 'RdYlGn', ['RGN']),
    '9': ('CVI', calculate_cvi, 'RdYlGn', ['RGN']),
    '10': ('CVI2', calculate_cvi2, 'RdYlGn', ['RGN']),
    '11': ('PRI', calculate_pri, 'Cividis', ['RGN']),
    '12': ('NDWI', calculate_ndwi, 'Blues', ['RGN', 'NGB']),
    '13': ('VARI', calculate_vari, 'Jet', ['RGB']),
    '14': ('EVI', calculate_evi, 'Plasma', ['RGN']),
    '15': ('NG', calculate_ng, 'Inferno', ['RGN']),
    '16': ('NGRDI', calculate_ngrdi, 'RdYlGn', ['RGB']),
    '17': ('ExG', calculate_exg, 'RdYlGn', ['RGB']),
    '18': ('ExR', calculate_exr, 'Magma', ['RGB']),
    '19': ('ExGR', calculate_exgr, 'Viridis', ['RGB']),
    '20': ('GLI', calculate_gli, 'RdYlGn', ['RGB']),
    '21': ('RGBVI', calculate_rgbvi, 'RdYlGn', ['RGB']),
    '22': ('TGI', calculate_tgi, 'Plasma', ['RGB']),
    '23': ('NGBDI', calculate_ngbdi, 'Blues', ['RGB']),
    '24': ('VDVI', calculate_vdvi, 'RdYlGn', ['RGB']),
    '25': ('MExG', calculate_mexg, 'Viridis', ['RGB']),
    '26': ('VEG', calculate_veg, 'Cividis', ['RGB']),
    '27': ('GCC', calculate_gcc, 'Inferno', ['RGB']),
    '28': ('CIVE', calculate_cive, 'Greys', ['RGB']),
    '29': ('NDTI', calculate_ndti, 'Blues', ['RGB']),
    '30': ('SCI', calculate_sci, 'Greys', ['RGB']),
    '31': ('NDBI', calculate_ndbi, 'Greys', ['RGB']),
    '32': ('BI', calculate_bi, 'Cividis', ['RGB']),
    '33': ('UI', calculate_ui, 'Greys', ['RGB']),
    '34': ('NDVI_Mod', calculate_ndvi_mod, 'RdYlGn', ['NGB']),
    '35': ('GNDVI', calculate_gndvi, 'RdYlGn', ['NGB']),
    '36': ('NDBI-Blue', calculate_ndbi_blue, 'Blues', ['NGB']),
    '37': ('NDGI', calculate_ndgi, 'RdYlGn', ['NGB']),
    '38': ('BGI', calculate_bgi, 'RdYlGn', ['NGB']),
    '39': ('EVI_Mod', calculate_evi_mod, 'Plasma', ['NGB']),
    '40': ('MSAVI', calculate_msavi, 'RdYlGn', ['NGB']),
    '41': ('ENDVI', calculate_endvi_ngb, 'Viridis', ['NGB']),
    '42': ('GNDWI', calculate_gndwi, 'Blues', ['NGB']),
    '43': ('CIG', calculate_cig, 'Cividis', ['NGB']),
    '44': ('GBNDVI', calculate_gbndvi, 'RdYlGn', ['NGB']),
    '45': ('GSAVI', calculate_gsavi, 'Magma', ['NGB']),
    '46': ('GRNDVI', calculate_grndvi, 'RdYlGn', ['NGB']),
    '48': ('GOSAVI', calculate_gosavi, 'Magma', ['NGB']),
    '49': ('NDWI', calculate_ndwi, 'Blues', ['NGB']),
    '50': ('BNDVI', calculate_bndvi, 'RdYlGn', ['NGB']),
    '51': ('NGBVI', calculate_ngbvi, 'RdYlGn', ['NGB']),
    '52': ('CIg', calculate_cig_simple, 'Cividis', ['NGB']),
    '53': ('BWDRVI', calculate_bwdrvi, 'RdYlGn', ['NGB']),
    '54': ('NDVI', calculate_ndvi, 'RdYlGn', ['RGN']),
    '55': ('GNDVI', calculate_gndvi, 'RdYlGn', ['RGN']),
    '56': ('SAVI', calculate_savi, 'Magma', ['RGN']),
    '57': ('MSAVI', calculate_msavi, 'RdYlGn', ['RGN']),
    '58': ('OSAVI', calculate_osavi, 'Magma', ['RGN']),
    '59': ('EVI2', calculate_evi2, 'Plasma', ['RGN']),
    '60': ('SR', calculate_sr, 'Cividis', ['RGN']),
    '61': ('RDVI', calculate_rdvi, 'RdYlGn', ['RGN']),
    '62': ('WDRVI', calculate_wdrvi, 'Inferno', ['RGN']),
    '63': ('MTVI2', calculate_mtvi2, 'Viridis', ['RGN']),
    '64': ('DVI', calculate_dvi, 'RdYlGn', ['RGN']),
    '65': ('NDWI', calculate_ndwi, 'Blues', ['RGN']),
    '66': ('CIg', calculate_cig, 'Cividis', ['RGN']),
    '67': ('CIred', calculate_cired, 'Cividis', ['RGN']),
    '68': ('CVI', calculate_cvi, 'RdYlGn', ['RGN']),
    '69': ('GRVI', calculate_grvi, 'RdYlGn', ['RGN']),
    '70': ('RGRI', calculate_rgri, 'RdYlGn', ['RGN']),
    '71': ('NGRDI', calculate_ngrdi, 'RdYlGn', ['RGN']),
    '72': ('GLI2', calculate_gli2, 'RdYlGn', ['RGB'])
}

# === Routes ===
@app.route('/')
def index():
    app.logger.debug('Index page accessed')
    return render_template('index.html', index_functions=index_functions, colormap_options=colormap_options)

@app.route('/process', methods=['POST'])
def process():
    try:
        if 'image' not in request.files:
            app.logger.error('No file part in request')
            return "No file part", 400
        file = request.files['image']
        if file.filename == '':
            app.logger.error('No selected file')
            return "No selected file", 400
        if file:
            index = request.form['index']
            image_type = request.form['image_type']
            colormap_name = request.form['colormap']
            label, func, default_colormap, valid_types = index_functions[index]

            if image_type not in valid_types:
                error_message = f"Error: The selected index {label} is not valid for the image type {image_type}."
                app.logger.error(error_message)
                return error_message, 400

            if colormap_name not in colormap_options:
                colormap_name = default_colormap

            filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filename)
            app.logger.debug(f"File saved to {filename}")

            rgb = imageio.imread(filename)

            # Read bands based on type
            if image_type == 'RGN':
                R = np.asarray(rgb[:,:,0], float)
                G = np.asarray(rgb[:,:,1], float)
                NIR = np.asarray(rgb[:,:,2], float)
                B = None
            elif image_type == 'NGB':
                NIR = np.asarray(rgb[:,:,0], float)
                G = np.asarray(rgb[:,:,1], float)
                B = np.asarray(rgb[:,:,2], float)
                R = None
            else:  # RGB
                R = np.asarray(rgb[:,:,0], float)
                G = np.asarray(rgb[:,:,1], float)
                B = np.asarray(rgb[:,:,2], float)
                NIR = None

            # Arguments per index
            args = {
                '1': (NIR, R),
                '2': (NIR, B),
                '3': (NIR, G, B),
                '4': (NIR, R, G),
                '5': (NIR, R),
                '6': (NIR, R, G),
                '7': (NIR if NIR is not None else R, G),
                '8': (NIR, R),
                '9': (NIR, R),
                '10': (NIR, R, G),
                '11': (NIR, R),
                '12': (G, NIR),
                '13': (G, R, B),
                '14': (NIR, R, B),
                '15': (G, NIR, R),
                '16': (G, R),
                '17': (G, R, B),
                '18': (R, G),
                '19': (G, R, B),
                '20': (G, R, B),
                '21': (G, R, B),
                '22': (R, G, B),
                '23': (G, B),
                '24': (G, R, B),
                '25': (G, R, B),
                '26': (G, R, B),
                '27': (G, R, B),
                '28': (R, G, B),
                '29': (R, G),
                '30': (R, G),
                '31': (R, G),
                '32': (R, G, B),
                '33': (R, B),
                '34': (NIR, G),
                '35': (NIR, G),
                '36': (NIR, B),
                '37': (G, B),
                '38': (G, B),
                '39': (NIR, G, B),
                '40': (NIR, G),
                '41': (NIR, G, B),
                '42': (G, NIR),
                '43': (NIR, G),
                '44': (NIR, G, B),
                '45': (NIR, G),
                '46': (NIR, G),
                '48': (NIR, G),
                '49': (G, NIR),
                '50': (NIR, B),
                '51': (G, B),
                '52': (NIR, G),
                '53': (NIR, B),
                '54': (NIR, R),
                '55': (NIR, G),
                '56': (NIR, R),
                '57': (NIR, R),
                '58': (NIR, R),
                '59': (NIR, R),
                '60': (NIR, R),
                '61': (NIR, R),
                '62': (NIR, R),
                '63': (NIR, R, G),
                '64': (NIR, R),
                '65': (G, NIR),
                '66': (NIR, G),
                '67': (NIR, R),
                '68': (NIR, R, G),
                '69': (G, R),
                '70': (R, G),
                '71': (G, R),
                '72': (G, R, B)
            }

            if index in args:
                index_args = args[index]
                if any(arg is None for arg in index_args):
                    error_message = f"Error: The selected index {label} requires bands that are not available in the image type {image_type}."
                    app.logger.error(error_message)
                    return error_message, 400
            else:
                error_message = f"Error: Invalid index selection."
                app.logger.error(error_message)
                return error_message, 400

            base_filename, ext = os.path.splitext(file.filename)
            output_name = os.path.join(app.config['RESULT_FOLDER'], f'{base_filename}_{label}.png')
            process_and_save(filename, output_name, func, label, colormap_name, *index_args)

            app.logger.debug(f'Processed image saved: {output_name}')
            return jsonify({'processed_image_url': url_for('static', filename=f'results/{base_filename}_{label}.png')})

    except Exception as e:
        app.logger.error(f"Error during processing: {e}")
        return "Internal server error", 500

if __name__ == '__main__':
    app.run(debug=True)
