package com.dvcs.handwriting;

import java.awt.Font;
import java.awt.GridLayout;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.MouseEvent;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;

import javax.swing.BoxLayout;
import javax.swing.JButton;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JProgressBar;
import javax.swing.JScrollPane;
import javax.swing.SwingConstants;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;

import com.dvcs.neuralnetwork.NeuralNetwork;
import com.dvcs.neuralnetwork.NeuralNetwork.ForwardPropagationResult;
import com.dvcs.tools.MatlabMatrixFactory;
import com.dvcs.tools.MatrixTools;

public class HandwritingExample {

	static final int NUM_CLASSES = 10;
	static final String[] classLabels = new String[] { "1", "2", "3", "4", "5",
			"6", "7", "8", "9", "0" };

	NeuralNetwork network;
	RealMatrix X;

	JFrame frame;
	HandwritingExampleApplet applet;
	JPanel sidebar;

	JProgressBar[] classBars;
	JLabel predictionLabel;

	public HandwritingExample() {
		buildNeuralNetwork();
	}

	public void init() {
		frame = new JFrame();
		frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		frame.setSize(710, 400);
		frame.getContentPane().setLayout(
				new BoxLayout(frame.getContentPane(), BoxLayout.X_AXIS));

		applet = new HandwritingExampleApplet();
		applet.init();
		applet.start();
		frame.add(applet);

		sidebar = initSidebar();

		frame.add(new JScrollPane(sidebar));
		frame.setVisible(true);

		// Show an example
		nextExample();
	}

	private JPanel initSidebar() {
		JPanel sidebar = new JPanel();

		GridLayout layout = new GridLayout(0, 2);
		sidebar.setLayout(layout);

		JButton nextButton = new JButton("Next example");
		nextButton.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e) {
				nextExample();
			}
		});
		sidebar.add(nextButton);

		predictionLabel = new JLabel();
		predictionLabel.setFont(new Font("Sans Serif", Font.PLAIN, 18));
		predictionLabel.setHorizontalAlignment(SwingConstants.CENTER);
		sidebar.add(predictionLabel);

		classBars = new JProgressBar[NUM_CLASSES];
		for (int i = 0; i < NUM_CLASSES; i++) {
			classBars[i] = new JProgressBar(0, 1000);

			sidebar.add(new JLabel(classLabels[i]));
			sidebar.add(classBars[i]);
		}

		return sidebar;
	}

	/**
	 * Display the result of a prediction.
	 */
	private void updateSidebar(double[] outputUnits, int predictedClass,
			long nanoseconds) {
		long ms = nanoseconds / 1000000L;
		predictionLabel
				.setText(classLabels[predictedClass] + " (" + ms + " ms)");

		for (int i = 0; i < outputUnits.length; i++) {
			classBars[i].setValue((int) (outputUnits[i] * 1000));
		}
	}

	private void buildNeuralNetwork() {
		RealMatrix Theta1 = null, Theta2 = null;

		try {
			X = getMatrix("X");
			Theta1 = getMatrix("Theta1");
			Theta2 = getMatrix("Theta2");
		} catch (Exception e) {
			e.printStackTrace();
		}

		network = new NeuralNetwork(Theta1, Theta2);
	}

	private void nextExample() {
		// Pick a random example
		int i = (int) Math.round(Math.random() * X.getRowDimension());
		double[] row = X.getRow(i);

		// Build a RealMatrix row vector
		RealMatrix x = new Array2DRowRealMatrix(row).transpose();

		// Benchmark
		long start = System.nanoTime();

		// Get the output layer (just a column vector)
		ForwardPropagationResult fResult = network.feedForward(x);
		double[] units = fResult.getA3().getColumn(0);
		int predictedClass = NeuralNetwork.maxIndex(units);
		long end = System.nanoTime();

		updateSidebar(units, predictedClass, end - start);

		// Show
		RealMatrix image = normalize(MatrixTools.reshape(row));
		applet.setImage(image);
	}

	private RealMatrix getMatrix(String matrixName) throws IOException,
			RuntimeException {
		String resourcePath = "com/dvcs/handwriting/resources";

		InputStream stream = getClass().getClassLoader().getResourceAsStream(
				resourcePath + "/" + matrixName + ".txt");

		if (stream == null) {
			throw new IOException("Matrix not found");
		}

		InputStreamReader reader = new InputStreamReader(stream);
		return MatlabMatrixFactory.loadFromReader(reader);
	}

	/**
	 * Normalize all elements of a matrix such that they lie in the range from 0
	 * to 1 (inclusive on both ends).
	 */
	private RealMatrix normalize(RealMatrix m) {
		m = m.copy();

		double max = Double.MIN_VALUE;
		double min = Double.MAX_VALUE;

		for (int i = 0; i < m.getRowDimension(); i++) {
			for (int j = 0; j < m.getRowDimension(); j++) {
				double value = m.getEntry(i, j);

				if (value > max) {
					max = value;
				} else if (value < min) {
					min = value;
				}
			}
		}

		for (int i = 0; i < m.getRowDimension(); i++) {
			for (int j = 0; j < m.getColumnDimension(); j++) {
				double value = m.getEntry(i, j);
				value = (value - min) / (max - min);
				if (value < 0) {
					System.out.println("--- fail");
				}

				m.setEntry(i, j, value);
			}
		}

		return m;
	}

	public void mouseEntered(MouseEvent e) {
	}

	public void mouseExited(MouseEvent e) {
	}

	public void mousePressed(MouseEvent e) {
	}

	public void mouseReleased(MouseEvent e) {
	}

	public void mouseClicked(MouseEvent e) {
		nextExample();
	}

	public static void main(String[] args) {
		new HandwritingExample().init();
	}

}