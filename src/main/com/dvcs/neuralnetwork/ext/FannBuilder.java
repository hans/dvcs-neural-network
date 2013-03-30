package com.dvcs.neuralnetwork.ext;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.List;
import java.util.logging.Logger;

import com.dvcs.neuralnetwork.Example;
import com.dvcs.neuralnetwork.NeuralNetworkBuilder;
import com.googlecode.fannj.ActivationFunction;
import com.googlecode.fannj.Fann;
import com.googlecode.fannj.Layer;
import com.googlecode.fannj.Trainer;

public class FannBuilder extends NeuralNetworkBuilder {

	private static final Logger LOGGER = Logger.getLogger("FANNNetworkBuilder");

	private static final int MAX_EPOCHS = 500000;
	private static final int EPOCHS_BETWEEN_REPORTS = 1000;
	private static final float DESIRED_ERROR = 0.001f;

	public Fann buildFann(int[] hiddenLayerSizes)
			throws InsufficientDataException {
		Fann network = buildFannStructure(hiddenLayerSizes);

		File temp;
		try {
			temp = File.createTempFile("dvcs_fann_", ".dat");
			
			FileOutputStream s = new FileOutputStream(temp);
			writeTrainingData(s);
			s.close();
		} catch ( IOException e ) {
			e.printStackTrace();
			return null;
		}

		Trainer trainer = new Trainer(network);
		float mse = trainer.train(temp.getPath(), MAX_EPOCHS,
				EPOCHS_BETWEEN_REPORTS, DESIRED_ERROR);

		return network;
	}

	private Fann buildFannStructure(int[] hiddenLayerSizes)
			throws InsufficientDataException {
		if ( !hasSufficientData() )
			throw new InsufficientDataException();

		List<Layer> layers = new ArrayList<Layer>();
		layers.add(Layer.create(inputLayerSize));

		for ( int i = 0; i < hiddenLayerSizes.length; i++ ) {
			layers.add(Layer.create(hiddenLayerSizes[i],
					ActivationFunction.FANN_SIGMOID_SYMMETRIC));
		}

		layers.add(Layer.create(outputLayerSize,
				ActivationFunction.FANN_SIGMOID_SYMMETRIC));

		return new Fann(layers);
	}

	private void writeTrainingData(OutputStream s) throws IOException {
		OutputStreamWriter w = new OutputStreamWriter(s);

		// First line: <# examples> <# inputs> <# outputs>
		w.write(examples.size() + " " + inputLayerSize + " " + outputLayerSize
				+ "\n");

		for ( Example ex : examples ) {
			float[] xs = ex.getX();
			for ( float x : xs ) {
				w.write(x + " ");
			}
			
			w.write("\n");
			
			float[] ys = ex.getY();
			for ( float y : ys ) {
				w.write(y + " ");
			}
			
			w.write("\n");
		}
		
		w.flush();
	}

}
