import logging
from flask import Flask, request, render_template, jsonify, send_from_directory, url_for
import os
import imageio.v2 as imageio
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

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

def create_colormap(cols):
    return LinearSegmentedColormap.from_list(name='custom1', colors=cols)

colormap_options = {
    'Color 1 (blue, green, yellow, red)': ['blue', 'green', 'yellow', 'red'],
    'Color 2 (gray, gray, red, yellow, green)': ['gray', 'gray', 'red', 'yellow', 'green'],
    'Color 3 (gray, blue, green, yellow, red)': ['gray', 'blue', 'green', 'yellow', 'red'],
    'Color 4 (black, gray, blue, green, yellow, red)': ['black', 'gray', 'blue', 'green', 'yellow', 'red']
}

def process_and_save(image_path, output_name, calculation_func, label, colormap_name, *args):
    try:
        colormap = colormap_options[colormap_name]
        rgb = imageio.imread(image_path)
        result = calculation_func(*args)

        abs_max = max(abs(np.min(result)), abs(np.max(result)))

        fig, ax = plt.subplots()
        image = ax.imshow(result, cmap=create_colormap(colormap), vmin=-abs_max, vmax=abs_max)
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
        cbar = plt.colorbar(image, cax=ax, orientation='horizontal', norm=norm)
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

        from PIL import Image
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
    '1': ('NDVI', calculate_ndvi, ['gray', 'blue', 'green', 'yellow', 'red'], ['RGN']),
    '2': ('NDVI_NGB', calculate_ndvi_ngb, ['gray', 'blue', 'green', 'yellow', 'red'], ['NGB']),
    '3': ('ENDVI', calculate_endvi, ['gray', 'blue', 'green', 'yellow', 'red'], ['NGB']),
    '4': ('ENDVI_RGN', calculate_endvi_rgn, ['gray', 'blue', 'green', 'yellow', 'red'], ['RGN']),
    '5': ('SAVI', calculate_savi, ['gray', 'blue', 'green', 'yellow', 'red'], ['RGN']),
    '6': ('TVI', calculate_tvi, ['gray', 'blue', 'green', 'yellow', 'red'], ['RGN']),
    '7': ('GNDVI', calculate_gndvi, ['gray', 'blue', 'green', 'yellow', 'red'], ['RGN', 'NGB', 'RGB']),
    '8': ('MSAVI', calculate_msavi, ['gray', 'blue', 'green', 'yellow', 'red'], ['RGN']),
    '9': ('CVI', calculate_cvi, ['gray', 'blue', 'green', 'yellow', 'red'], ['RGN']),
    '10': ('CVI2', calculate_cvi2, ['gray', 'blue', 'green', 'yellow', 'red'], ['RGN']),
    '11': ('PRI', calculate_pri, ['gray', 'blue', 'green', 'yellow', 'red'], ['RGN']),
    '12': ('NDWI', calculate_ndwi, ['gray', 'blue', 'green', 'yellow', 'red'], ['RGN', 'NGB']),
    '13': ('VARI', calculate_vari, ['gray', 'blue', 'green', 'yellow', 'red'], ['RGN', 'NGB', 'RGB']),
    '14': ('EVI', calculate_evi, ['gray', 'blue', 'green', 'yellow', 'red'], ['RGN']),
    '15': ('NG', calculate_ng, ['gray', 'blue', 'green', 'yellow', 'red'], ['RGN'])
}

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
            label, func, colormap, valid_types = index_functions[index]

            if image_type not in valid_types:
                error_message = f"Error: The selected index {label} is not valid for the image type {image_type}."
                app.logger.error(error_message)
                return error_message, 400

            filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filename)
            app.logger.debug(f"File saved to {filename}")

            rgb = imageio.imread(filename)

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

            args = {
                '1': (NIR, R),
                '2': (NIR, B),
                '3': (NIR, G, B),
                '4': (NIR, R, G),
                '5': (NIR, R),
                '6': (NIR, R, G),
                '7': (NIR, G),
                '8': (NIR, R),
                '9': (NIR, R),
                '10': (NIR, R, G),
                '11': (NIR, R),
                '12': (G, NIR),
                '13': (G, R, B),
                '14': (NIR, R, B),
                '15': (G, NIR, R)
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
