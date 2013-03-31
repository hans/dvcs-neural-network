package com.dvcs.neuralnetwork.driver;

import java.awt.Color;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;

import javax.swing.JApplet;

import org.jblas.DoubleMatrix;

public class MatrixImageApplet extends JApplet {

	DoubleMatrix image;
	BufferedImage offscreen;

	static final int SCALE = 2;

	public void setImage(DoubleMatrix i) {
		image = i.transpose();

		int width = image.getColumns() * SCALE;
		int height = image.getRows() * SCALE;
		setSize(width, height);

		offscreen = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);

		repaint();
	}

	@Override
	public void paint(Graphics g) {
		if ( offscreen == null )
			return;
		
		paint((Graphics2D) offscreen.getGraphics());
		g.drawImage(offscreen, 0, 0, null);
	}
	
	private void paint(Graphics2D g) {
		if ( image == null )
			return;

		for ( int i = 0; i < image.getRows(); i++ ) {
			for ( int j = 0; j < image.getColumns(); j++ ) {
				float intensity = (float) image.get(i, j);
				Color c = new Color(intensity, intensity, intensity);

				g.setColor(c);
				g.fillRect(i * SCALE, j * SCALE, SCALE, SCALE);
			}
		}
	}

}
