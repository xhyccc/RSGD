package edu.mst.xionglab.rsgd;

import Jama.Matrix;

public interface ObjectiveFunction {
	public double evaluation(Matrix input);
	public Matrix gradient(Matrix input, Matrix prev);
}
