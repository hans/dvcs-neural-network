package com.dvcs.neuralnetwork.driver;

import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.util.logging.Logger;

import javax.imageio.ImageIO;

import org.jblas.DoubleMatrix;
import org.jblas.util.Random;

import com.dvcs.neuralnetwork.NeuralNetwork;
import com.dvcs.neuralnetwork.NeuralNetwork.ForwardPropagationResult;
import com.dvcs.neuralnetwork.NeuralNetworkBuilder;
import com.dvcs.neuralnetwork.NeuralNetworkBuilder.DimensionMismatchException;
import com.dvcs.neuralnetwork.NeuralNetworkBuilder.Example;
import com.dvcs.neuralnetwork.NeuralNetworkBuilder.InsufficientDataException;
import com.dvcs.neuralnetwork.driver.DataQueueListener.NewDataCallback;

public class Driver {
	static final String QUEUE_NAME = "robotData";
	static final double LAMBDA = 0.75;
	static final int HIDDEN_LAYER_UNITS = 100;
	static final int OUTPUT_LAYER_UNITS = 10;

	private static final Logger LOGGER = Logger
			.getLogger("NeuralNetworkDriver");

	private DriverGUI gui;
	private NeuralNetwork network;
	private DataCollector collector;
	private DataCollector predictor;
	private NeuralNetworkBuilder builder;

	private NewDataCallback dataCollectorCallback = new NewDataCallback() {
		public void receivedData(byte[] data) {
			DoubleMatrix m = parseImageData(data);

			double[] x = m.toArray();

			// TODO: y
			double[] y = new double[] { 1, 0, 0, 0, 0, 0, 0, 0, 0, 0 };

			Example ex = builder.new Example(x, y);
			try {
				builder.addExample(ex);
			} catch ( DimensionMismatchException e ) {
				e.printStackTrace();
			}

			// Display image
			gui.loadImageMatrix(m);
		}
	};

	private NewDataCallback dataPredictorCallback = new NewDataCallback() {
		public void receivedData(byte[] data) {
			if ( network == null ) {
				LOGGER.severe("Driver asked to make predictions before network"
						+ "was built");
			}

			DoubleMatrix m = parseImageData(data);

			// Build a row vector
			int length = m.getRows() * m.getColumns();
			DoubleMatrix x = new DoubleMatrix(1, length, m.toArray());

			long start = System.nanoTime();

			ForwardPropagationResult fResult = network.feedForward(x);
			double[] outputUnits = fResult.getOutputLayer().getColumn(0)
					.toArray();
			int predictedClass = NeuralNetwork.maxIndex(outputUnits);

			long end = System.nanoTime();

			gui.loadImageMatrix(m);
			gui.displayPropagationResult(outputUnits, predictedClass, start
					- end);
		}
	};

	public Driver(DriverGUI _gui) {
		gui = _gui;
		builder = new NeuralNetworkBuilder();
		collector = new DataCollector(QUEUE_NAME, dataCollectorCallback);
		predictor = new DataCollector(QUEUE_NAME, dataPredictorCallback);
	}

	public boolean isCollecting() {
		return collector.isListening();
	}

	public void startCollecting() {
		LOGGER.info("Beginning data collection");

		collector.startQueueListener();
	}

	public void stopCollecting() {
		LOGGER.info("Ending data collection");

		collector.stopQueueListener();
	}

	/**
	 * @return Whether the builder has enough data to create a neural network
	 */
	public boolean hasSufficientData() {
		return builder.hasSufficientData();
	}

	/**
	 * Train the neural network using the given sample data.
	 */
	public void trainNeuralNetwork() {
		try {
			network = builder.buildNetwork(new int[] { HIDDEN_LAYER_UNITS },
					LAMBDA);
		} catch ( InsufficientDataException e ) {
			e.printStackTrace();
		}
	}

	public void startFeedForward() {
		LOGGER.info("Beginning feedforward processing");

		predictor.startQueueListener();
	}
	
	public void stopFeedForward() {
		LOGGER.info("Ending feedforward processing");
		
		predictor.stopQueueListener();
	}

	private DoubleMatrix parseImageData(byte[] data) {
		BufferedImage im = null;
		try {
			im = ImageIO.read(new ByteArrayInputStream(data));
		} catch ( IOException e ) {
			e.printStackTrace();
			return null;
		}

		if ( im == null ) {
			LOGGER.severe("Failed to parse data received from queue. "
					+ "Only valid raw image data should be published on "
					+ "this queue.");
			return null;
		}

		DoubleMatrix m = ImageConverter.convertImageToMatrix(im, true);
		m = ImageConverter.normalize(m);

		return m;
	}

	public class OutputProvider {

	}
}