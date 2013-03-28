package com.dvcs.neuralnetwork;

import java.util.logging.Logger;

import org.encog.engine.network.activation.ActivationSigmoid;
import org.encog.mathutil.randomize.ConsistentRandomizer;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.basic.BasicMLDataSet;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.layers.BasicLayer;
import org.encog.neural.networks.training.propagation.back.Backpropagation;

public class EncogNetworkBuilder extends NeuralNetworkBuilder {

	private static final Logger LOGGER = Logger.getLogger("EncogNetworkBuilder");
	
	public BasicNetwork buildEncogNetwork(int[] hiddenLayerSizes,
			double learnRate, double momentum) throws InsufficientDataException {
		BasicNetwork network = buildEncogStructure(hiddenLayerSizes);

		// Randomize weights of the network
		new ConsistentRandomizer(-1, 1, 500).randomize(network);

		// Convert Example array to a format Encog likes
		double[][] xs = new double[examples.size()][inputLayerSize];
		double[][] ys = new double[examples.size()][outputLayerSize];

		for (int i = 0; i < examples.size(); i++) {
			Example ex = examples.get(i);
			xs[i] = ex.getX();
			ys[i] = ex.getY();
		}

		MLDataSet trainingSet = new BasicMLDataSet(xs, ys);

		Backpropagation train = new Backpropagation(network, trainingSet,
				learnRate, momentum);
		
		int epoch = 1;
		do {
			train.iteration();
			
			LOGGER.info("Epoch #" + epoch + "\tError: " + train.getError());
			epoch++;
		} while ( train.getError() > 0.01 );
		
		return network;
	}

	private BasicNetwork buildEncogStructure(int[] hiddenLayerSizes)
			throws InsufficientDataException {
		if (!hasSufficientData())
			throw new InsufficientDataException();

		BasicNetwork network = new BasicNetwork();
		network.addLayer(new BasicLayer(null, true, inputLayerSize));

		for (int i = 0; i < hiddenLayerSizes.length; i++) {
			network.addLayer(new BasicLayer(new ActivationSigmoid(), true,
					hiddenLayerSizes[i]));
		}

		network.addLayer(new BasicLayer(new ActivationSigmoid(), false,
				outputLayerSize));
		network.getStructure().finalizeStructure();
		network.reset();

		return network;
	}

}
