/*package pll.layers

import pll.initializers.Initializer
import pll.utils.RichArray._

class Conv2D(
  val in_height:Int,
  val in_width:Int,
  val in_channels:Int,
  val out_channels:Int,
  val ksize:Int,
  val stride:Int = 1,
  val k_initer:Initializer = Initializer.normal(0.01f)
) extends Layer {

  val out_height = (in_height - ksize) / stride + 1
  val out_width = (in_width - ksize) / stride + 1
 
  val kernels = k_initer(ksize*ksize*in_channels*out_channels)
  val grad = Array.ofDim[Float](kernels.size)

  var xs = List[Array[Float]]()
  val N = out_height * out_width
  val M = out_channels
  val K = ksize * ksize * in_channels

  def forward(x:Array[Float]) = {
    val features = this.convert(x)
    xs ::= features
    val output = Array.ofDim[Float](N*M)

    blas.BLAS.matmul(features, kernels, output, N, M, K)
    output.transpose(N, M)
  }

  def forward(xs:Array[Array[Float]]) = {
    val features = xs.map(this.convert)
    features.foreach(f => this.xs ::= f)
    val output = Array.ofDim[Float](N*M*xs.size)

    blas.BLAS.matmul(features.flatten, kernels, output, N*xs.size, M, K)

    output.grouped(N*M).map(_.transpose(N,M)).toArray
  }

  def backward(d:Array[Float]) = {
    val dloss = d.transpose(M, N)

    // backward kernel
    val past_x = xs.head
    xs = xs.tail
    val dkernels = Array.ofDim[Float](K * M)
    blas.BLAS.matmul(past_x.transpose(N, K), dloss, dkernels, K, M, N)
    Range(0, grad.size).foreach(i => grad(i) += dkernels(i))
    // backward input
    val dinput = Array.ofDim[Float](N * K)
    blas.BLAS.matmul(dloss, kernels.transpose(K, M), dinput, N, K, M)

    this.reconvert(dinput)
  }

  def backward(ds:Array[Array[Float]]) = {
    val dloss = ds.map(_.transpose(M, N)).flatten

    // backward kernel
    val past_xs = xs.take(ds.size).reverse.toArray.flatten
    xs = xs.drop(ds.size)
    val dkernels = Array.ofDim[Float](K * M)
    blas.BLAS.matmul(past_xs.transpose(N*ds.size, K), dloss, dkernels, K, M, N*ds.size)
    Range(0, grad.size).foreach(i => grad(i) += dkernels(i))

    // backward input
    
    val dinput = Array.ofDim[Float](N*ds.size*K)
    blas.BLAS.matmul(dloss, kernels.transpose(K, M), dinput, N*ds.size, K, M)

    dinput.grouped(N*K).map(this.reconvert).toArray
  }

  def reset() {
    Range(0, grad.size).foreach(i => grad(i) = 0)
    xs = Nil
  }

  override def paramWithGrad = {
    List((kernels, grad))
  }

  override def params = {
    List(kernels)
  }

  def load(k:Array[Float]) {
    assert(kernels.size == k.size)
    Range(0, k.size).foreach(i => kernels(i) = k(i))
  }

  def convert(x:Array[Float]) = {
    val f_height = out_height * out_width
    val f_width = ksize * ksize * in_channels
    val features = Array.ofDim[Float](f_height*f_width)

    def findex(i:Int, j:Int, k:Int, l:Int, m:Int) = 
      i*f_width*out_width+j*f_width+k*ksize*ksize+l*ksize+m

    def xindex(i:Int, j:Int, k:Int, l:Int, m:Int) = 
      i*in_width+j+k*in_width*in_height+l*in_width+m

    for(
      i <- 0 until out_height by stride;
      j <- 0 until out_width by stride;
      k <- 0 until in_channels;
      l <- 0 until ksize;
      m <- 0 until ksize
    ) features(findex(i,j,k,l,m)) = x(xindex(i,j,k,l,m))
    features
  }

  def reconvert(x:Array[Float]) = {
    val f_height = out_height * out_width
    val f_width = ksize * ksize * in_channels
    val features = Array.ofDim[Float](in_height*in_width*in_channels)

    def findex(i:Int, j:Int, k:Int, l:Int, m:Int) =
      i*f_width*out_width+j*f_width+k*ksize*ksize+l*ksize+m

    def xindex(i:Int, j:Int, k:Int, l:Int, m:Int) = 
      i*in_width+j+k*in_width*in_height+l*in_width+m

    for(
      i <- 0 until out_height by stride;
      j <- 0 until out_width by stride;
      k <- 0 until in_channels;
      l <- 0 until ksize;
      m <- 0 until ksize
    ) features(xindex(i,j,k,l,m)) += x(findex(i,j,k,l,m))

    features
  }

}
 */
