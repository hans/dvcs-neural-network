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
import javax.swing.SwingUtilities;
import javax.swing.Timer;

import org.jblas.DoubleMatrix;

import com.dvcs.neuralnetwork.NeuralNetwork;
import com.dvcs.neuralnetwork.NeuralNetwork.ForwardPropagationResult;
import com.dvcs.tools.MatlabMatrixFactory;
import com.dvcs.tools.MatrixTools;

import de.jungblut.math.DoubleVector;
import de.jungblut.math.minimize.MinimizerListener;

public class HandwritingExample {

	static final int NUM_CLASSES = 10;
	static final String[] classLabels = new String[] { "1", "2", "3", "4", "5",
			"6", "7", "8", "9", "0" };

	static final double LAMBDA = 0.75;

	NeuralNetwork network;
	DoubleMatrix X;
	DoubleMatrix Y;

	JFrame frame;
	HandwritingExampleApplet applet;

	JPanel sidebar;
	JProgressBar[] classBars;
	JLabel predictionLabel;

	JPanel trainingSidebar;
	JLabel costLabel;

	ActionListener timerAction = new ActionListener() {
		public void actionPerformed(ActionEvent e) {
			nextExample();
		}
	};
	Timer timer;

	public HandwritingExample() {
		buildNeuralNetwork();
	}

	public void init() {
		frame = new JFrame();
		frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		frame.setSize(1010, 400);
		frame.getContentPane().setLayout(
				new BoxLayout(frame.getContentPane(), BoxLayout.X_AXIS));

		applet = new HandwritingExampleApplet();
		applet.init();
		applet.start();
		frame.add(applet);

		sidebar = initSidebar();
		trainingSidebar = initTrainingSidebar();

		frame.add(new JScrollPane(sidebar));
		frame.add(trainingSidebar);
		frame.setVisible(true);

		timer = new Timer(500, timerAction);

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

		JButton cycleButton = new JButton("Cycle examples");
		cycleButton.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e) {
				cycleExamples();
			}
		});
		sidebar.add(cycleButton);

		predictionLabel = new JLabel();
		predictionLabel.setFont(new Font("Sans Serif", Font.PLAIN, 18));
		predictionLabel.setHorizontalAlignment(SwingConstants.CENTER);
		sidebar.add(predictionLabel);

		// hack
		sidebar.add(new JLabel(""));

		classBars = new JProgressBar[NUM_CLASSES];
		for (int i = 0; i < NUM_CLASSES; i++) {
			classBars[i] = new JProgressBar(0, 1000);

			sidebar.add(new JLabel(classLabels[i]));
			sidebar.add(classBars[i]);
		}

		return sidebar;
	}

	private JPanel initTrainingSidebar() {
		JPanel sidebar = new JPanel();
		sidebar.setLayout(new GridLayout(0, 1));

		final MinimizerListener listener = new MinimizerListener() {
			public void minimizationIterationFinished(final int n,
					final double cost, DoubleVector parameters) {
				SwingUtilities.invokeLater(new Runnable() {
					public void run() {
						costLabel.setText("Iteration " + n + "\n" + "Cost: "
								+ String.format("%.04f", cost));
					}
				});
			}
		};

		final JButton trainButton = new JButton("Train");
		trainButton.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e) {
				trainButton.setEnabled(false);

				new Thread() {
					public void run() {
						network.train(X, Y, NUM_CLASSES, LAMBDA, listener);

						SwingUtilities.invokeLater(new Runnable() {
							public void run() {
								trainButton.setEnabled(true);
							}
						});
					}
				}.start();
			}
		});
		sidebar.add(trainButton);

		costLabel = new JLabel("Cost: 0.0000");
		sidebar.add(costLabel);

		return sidebar;
	}

	/**
	 * Display the result of a prediction.
	 */
	private void updateSidebar(double[] outputUnits, int predictedClass,
			long nanoseconds) {
		long ms = nanoseconds / 1000000L;
		predictionLabel.setText(classLabels[predictedClass] + " (" + ms
				+ " ms)");

		for (int i = 0; i < outputUnits.length; i++) {
			classBars[i].setValue((int) (outputUnits[i] * 1000));
		}
	}

	private void buildNeuralNetwork() {
		DoubleMatrix Theta1 = null, Theta2 = null;

		try {
			Theta1 = getMatrix("Theta1");
			X = getMatrix("X");
			Theta2 = getMatrix("Theta2");
			Y = getMatrix("Y");
		} catch (Exception e) {
			e.printStackTrace();
		}

		network = new NeuralNetwork(Theta1, Theta2);
	}

	private void cycleExamples() {
		timer.start();
	}

	private void nextExample() {
		// Pick a random example
		int i = (int) Math.round(Math.random() * X.getRows());

		// Build a DoubleMatrix row vector
		DoubleMatrix x = X.getRow(i);

		// Benchmark
		long start = System.nanoTime();

		// Get the output layer (just a column vector)
		ForwardPropagationResult fResult = network.feedForward(x);
		double[] units = fResult.getA3().getColumn(0).toArray();
		int predictedClass = NeuralNetwork.maxIndex(units);
		long end = System.nanoTime();

		updateSidebar(units, predictedClass, end - start);

		// Show
		DoubleMatrix image = normalize(MatrixTools.reshape(x.toArray()));
		applet.setImage(image);
	}

	private DoubleMatrix getMatrix(String matrixName) throws IOException,
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
	private DoubleMatrix normalize(DoubleMatrix m) {
		m = MatrixTools.copy(m);

		double max = Double.MIN_VALUE;
		double min = Double.MAX_VALUE;

		for (int i = 0; i < m.getRows(); i++) {
			for (int j = 0; j < m.getColumns(); j++) {
				double value = m.get(i, j);

				if (value > max) {
					max = value;
				} else if (value < min) {
					min = value;
				}
			}
		}

		for (int i = 0; i < m.getRows(); i++) {
			for (int j = 0; j < m.getColumns(); j++) {
				double value = m.get(i, j);
				value = (value - min) / (max - min);
				if (value < 0) {
					System.out.println("--- fail");
				}

				m.put(i, j, value);
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
