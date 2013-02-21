package com.dvcs.handwriting;

import java.awt.Color;
import java.awt.Graphics;
import java.awt.GridLayout;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.Arrays;

import javax.swing.JApplet;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;

import com.dvcs.tools.MatlabMatrixFactory;

public class HandwritingExampleApplet extends JApplet {

	RealMatrix image;

	static final int SCALE = 20;

	public void setImage(RealMatrix i) {
		image = i;
		setSize(image.getRowDimension() * SCALE, image.getColumnDimension()
				* SCALE);
		
		repaint();
	}

	@Override
	public void paint(Graphics g) {
		for (int i = 0; i < image.getRowDimension(); i++) {
			for (int j = 0; j < image.getColumnDimension(); j++) {
				float intensity = (float) image.getEntry(i, j);
				Color c = new Color(intensity, intensity, intensity);

				g.setColor(c);
				g.fillRect(i * SCALE, j * SCALE, SCALE, SCALE);
			}
		}
	}
	
}
