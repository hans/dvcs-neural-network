package com.dvcs.neuralnetwork.driver;

import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.util.logging.Logger;

import javax.imageio.ImageIO;

import org.jblas.DoubleMatrix;
import org.jblas.util.Random;

import com.dvcs.neuralnetwork.NeuralNetwork;
import com.dvcs.neuralnetwork.NeuralNetworkBuilder;
import com.dvcs.neuralnetwork.NeuralNetworkBuilder.DimensionMismatchException;
import com.dvcs.neuralnetwork.NeuralNetworkBuilder.Example;
import com.dvcs.neuralnetwork.driver.DataQueueListener.NewDataCallback;

public class NeuralNetworkDriver {
	static final String QUEUE_NAME = "robotData";
	static final double LAMBDA = 0.75;
	static final int HIDDEN_LAYER_UNITS = 100;
	static final int OUTPUT_LAYER_UNITS = 10;

	private static final Logger LOGGER = Logger
			.getLogger("NeuralNetworkDriver");

	private NeuralNetworkDriverGUI gui;
	private NeuralNetwork network;
	private NeuralNetworkDataCollector collector;
	private NeuralNetworkBuilder builder;

	private NewDataCallback dataCallback = new NewDataCallback() {
		public void receivedData(byte[] data) {
			BufferedImage im = null;
			try {
				im = ImageIO.read(new ByteArrayInputStream(data));
			} catch ( IOException e ) {
				e.printStackTrace();
				return;
			}

			if ( im == null ) {
				LOGGER.severe("Failed to parse data received from queue. "
						+ "Only valid raw image data should be published on "
						+ "this queue.");
				return;
			}

			DoubleMatrix m = ImageConverter.convertImageToMatrix(im, true);
			m = ImageConverter.normalize(m);

			double[] x = m.toArray();

			// TODO: y
			Random r = new Random();
			double[] y = new double[] { r.nextDouble(), r.nextDouble() };

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

	public NeuralNetworkDriver(NeuralNetworkDriverGUI _gui) {
		gui = _gui;
		builder = new NeuralNetworkBuilder();
		collector = new NeuralNetworkDataCollector(QUEUE_NAME, dataCallback);
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
		// Generate X and Y matrices from examples
	}

	public class OutputProvider {

	}
}