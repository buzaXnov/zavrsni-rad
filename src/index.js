import * as tf from '../node_modules/@tensorflow/tfjs'
import { loadGraphModel } from '../node_modules/@tensorflow/tfjs-converter'
import labels from './labels.json'

// const MODEL_URL = 'https://storage.cloud.google.com/mobile-retrained-model-zavrsni/model/model.json?folder=true&organizationId=true ';
// const MODEL_URL = 'https://00e9e64bace35696901fac9e967823f03dc666db2b76bf658d-apidata.googleusercontent.com/download/storage/v1/b/mobile-retrained-model-zavrsni/o/model%2Fmodel.json?qk=AD5uMEvdCVkf3o7lky0QNFemTBY4QQcn9MRDMLQMAKp5T1eHBiIaXQkpkPOIu2MEILJHf64DBkaCDVtJ09k3RmdqNRNPc0h7dYLCDvCn3HSYT8_MmKxSVs_z57GmDd1SXpP0Om4vLwrqNDcOi0nY7MfpefALL0e9L2JrLv3n424JIL-TIfCDLvhgHHDCKR7vUhXhTS7wh-KdQIFYljBL9KB8TJ45jjZGb6R1Y-5sik22K2KZwLNp3AmAfNLmLSL-mdmdLvvE0-W8lCFmCf5vBtxTVPQWcfYzarrOOZHjSAPKcn_3ATzpaEJXD0Dj7IOUKdDtYSnatPeS38I4lrQ9mD3-9QCfRlUDwfqNHfe5RA7E9G62hufnQeqeJA402dd-Ne6Wms264BCrpKV-leel8uXmTcigiupm3nGxoev09BqRbU32JmSRWHXY1IZ4HH-E-_dogEvwNSHV_z-M7m7X0o3rwFeoK93I2qn0RedMMp8_dincpQy9xU7bidSTzuuPjrlhQyYtFhlpnZP3QOWVsyr_UFdND3_fKOmsLA3i4UHIOazTwa6aYwurZdMh4Gd4DUqzZtHmNO6Ww-P685Ii8F-q9q1EFMV8NJYB6ULa3aRbROwA2NYd_yihJA-nDAsD4OfvTTfCv5qzWgqH4aUWxzQ_kNpL_-ODAbjUFvd7BjhhHdSUxHCIP-ZU0SLRnTct90Yo9qxaR01y7Icnfmx5341OBficy4NnvIT9lFhUYDRGJYCKyNxy78LnjXwzMhE7udl9wlMLQ0GQJ40Gd-4qXGuYVBNO6cMdEjiDKM_KXcR-3TsEchzH4hQ';
// const MODEL_URL = 'http://kamis-stud01.fesb.hr/home/fbutic19/model.json';
const MODEL_URL = 'https://github.com/buzaXnov/zavrsni-rad/blob/master/model.json';

const IMAGE_SIZE = 224;

const load_model = async () => {
	console.log("Loading model...");
	const model = await loadGraphModel(MODEL_URL);

	// model.predict({ Placeholder: input })

	console.log("Model loaded.");
	return model;
}

const predict = async (img, model) => {
	const t0 = performance.now();
	const image = tf.browser.fromPixels(img).toFloat();
	const resized = tf.image.resizeBilinear(image, [IMAGE_SIZE, IMAGE_SIZE]);
	const offset = tf.scalar(255 / 2);
	const normalized = resized.sub(offset).div(offset);

	const input = normalized.expandDims(0);
	const output = await tf.tidy(() => model.predict({ Placeholder: input })).data()

	const predictions = labels
		.map((label, index) => ({ label, accuracy: output[index] }))
		.sort((a, b) => b.accuracy - a.accuracy);

	const time = `${performance.now() - t0}.toFixed(1) ms`
	return { predictions, time };
}

const start = async () => {
	const input = document.getElementById('input');
	const output = document.getElementById('output');
	const model = await load_model();

	const predictions = await predict(input, model);
	output.append(JSON.stringify(predictions, null, 2));
}

start()
