package edu.mst.xionglab.rsgd.funcs;

import Jama.Matrix;
import edu.mst.xionglab.rsgd.SingleVariableGradientDescent;
import edu.mst.xionglab.rsgd.SingleVariableObjectiveFunction;

public class QuadraticFunction implements SingleVariableObjectiveFunction {
	private Matrix _A, _B; // A is a p\times p matrix, B is a p \times 1 matrix
	private double _C; // c is a scalar

	public QuadraticFunction(Matrix A, Matrix B, double C) {
		this._A=A;
		this._B=B;
		this._C=C;
	}
	

	@Override
	public Matrix gradient(Matrix input, Matrix prev) {
		// TODO Auto-generated method stub
		return this.gradient(input);
	}

	@Override
	public double evaluation(Matrix input) {
		// TODO Auto-generated method stub
		return input.transpose().times(this._A).times(input).plus(input.transpose().times(_B)).get(0, 0)+_C;
	}


	public Matrix gradient(Matrix input) {
		// TODO Auto-generated method stub
		return this._A.times(2).times(input).plus(_B);
	}

	public static void main(String[] args){
		double[][] aMat =new double[100][100];
		for(int i=0;i<aMat.length;i++){
			for(int j=0;j<aMat[i].length;j++){
				aMat[i][j] = Math.pow(0.8,Math.abs(i-j));
			}
		}
		double[][] bMat = new double[100][1];
		double[][] pMat = new double[100][1];
		for(int i=0;i<bMat.length;i++){
			bMat[i][0]=-1*Math.random();
			pMat[i][0]=0;//Math.random();
		}
		Matrix A =new Matrix(aMat);
		Matrix B = new Matrix(bMat);
		Matrix parameter = new Matrix(pMat);
		
		QuadraticFunction aFunc =new QuadraticFunction(A,B,0);
		SingleVariableGradientDescent gdi = new SingleVariableGradientDescent(aFunc, parameter);
		gdi.GDProcedure();
		System.out.println(gdi.getResult().getColumnDimension()+"\t"+gdi.getResult().getRowDimension());
		
	}

	
}
