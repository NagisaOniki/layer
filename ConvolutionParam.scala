package layer
import breeze.linalg._
import pll.utils.RichArray._
class ConvolutionParam(
  val KW:Int,
  val IH:Int,
  val IW:Int,
  val IC:Int,
  val OC:Int,
  val ep:Float,
  val p1:Float
) extends Layer {
  override type T = Float
  //I:入力、O:出力、H:height、W:width、IC:inputチャネル数、OC:outputチャネル数
  val OH = IH - KW + 1
  val OW = IW - KW + 1
  val rand = new util.Random(0)
  var xs = List[Array[Float]]()
  var K = Array.ofDim[T](OC,IC,KW,KW)
  for(i<-0 until OC ; j<-0 until IC ; k<-0 until KW ; l<-0 until KW){
    K(i)(j)(k)(l) = rand.nextGaussian().toFloat *0.01f // / math.sqrt(KW*KW*IC).toFloat // = *0.1f > *0.01f (精度比較)
  }
  val grad = Array.ofDim[Float](K.size)
  var K_save = Array.ofDim[T](OC,IC,KW,KW)
  var dK = Array.ofDim[T](OC,IC,KW,KW)
  var Vtemp = Array[T]()
  var s = Array.ofDim[T](OC,IC,KW,KW)
  var r = Array.ofDim[T](OC,IC,KW,KW)
  var p1t = 1f //updataで使うp1のt乗
  var p2t = 1f //updataで使うp2のt乗
  var bc = 0f
  def forward(V:Array[T]) = {
    Vtemp = V
    var Z = Array.ofDim[T](OC * OH * OW)
    for(i<-0 until OC ; j<-0 until OH ; k<-0 until OW){
      for(l<-0 until IC ; m<-0 until KW ; n<-0 until KW){
        Z(i*OH*OW + j*OW+k) += V(l*IH*IW + (j+m)*IW+(k+n)) * K(i)(l)(m)(n)
      }
    }
    Z
  }

  override def forward(V:Array[Array[T]])={
    V.map(forward)
  }

/* def forward(xs:Array[Array[T]])={
    val features = xs.map(this.convert)
    features.foreach(f => this.xs ::= f)
    val output = Array.ofDim[Float](OH*OW*xs.size)

    blas.BLAS.matmulF(features.flatten, K.flatten.flatten.flatten, output, OH*OW*xs.size, OC, KW*KW*IC)

    output.grouped(OH*OW*OC).map(_.transpose(OH*OW,OC)).toArray
  }*/


/*
     override def forward(V:Array[Array[T]])={
   var X = Array.ofDim[T](OH*OW , KW*KW*IC) //Vを並び替えたもの
   var Xt = Array.ofDim[T](KW*KW*IC , OH*OW) //Vの転置
   var Ft = Array.ofDim[T](OC , IC*KW*KW) //Kの転置
   var Y = Array.ofDim[T](OC , OH , OW) //yをzに並べ替える
   var Yt = Array.ofDim[T](OC , OH*OW)
   //V->X->Xt

   var row = 0
   for(n <-0 until V.size){
   for(i<-0 until IC ; j<-0 until OH ; k<-0 until OW){
   for(l<-0 until KW ; m<-0 until KW){
   X(row)(l*KW+m) = V(n)(i*IH*IW + j*IW + k+ l*IW +m)
   print(X(row)(l*KW+m)+",")
 }
   row += 1
   }
   }

   for(c<-0 until IC){
   for(i<-0 until OH){
   for(j<-0 until OW){
   for(p<-0 until KW){ //KH
   for(q<-0 until KW){
   X(i*OW+j)(c*KW*KW + p*KW + q) = V(c*IH*IW )( i+p + (j+q)*IW)
   print(X(i*OW+j)(c*KW*KW + p*KW + q)+",")
   }
   }
   }
   }
   }
   for(i<-0 until OH*OW ; j<-0 until KW*KW*IC){
   print(X(i)(j)+",")
   }
   println("")
   Xt = DenseMatrix(X).toArray
   for(i<-0 until KW*KW*IC ; j<-0 until OH*OW){
   print(Xt(i)(j)+",")
   }
   println("")
   //K->Ft
   val Ftemp = ((K.flatten).flatten).flatten
   var index = 0
   for(i<-0 until OC){
   for(j<-0 until IC*KW*KW){
   Ft(i)(j) = Ftemp(index)
   }
   }
   for(i<-0 until OC ; j<-0 until IC*KW*KW){
   for(p<-0 until OC ; q<-0 until IC ; r<-0 until KW ; s<-0 until KW){
   Ft(i)(j) = K(p)(q)(r)(s)
   }
   }
   blas.BLAS.matmulF(Ft.flatten,Xt.flatten,Yt.flatten, OC , OH*OW , IC*KW*KW)//A,B,C(AxB=C),M,N,K
   //Ft*Xt=Yt,,,Ft:OCxIC*KW*KW,,,Xt:IC*KW*KWxOH*OW,,,Yt:OCxOH*OW
   for(i<-0 until OC){
   for(p<-0 until OH ; q<-0 until OW){
   Y(i)(p)(q) == Yt(i)(p*OW+q)
   }
   }
   Y.flatten
 }
*/

  def backward(G:Array[T]) = {
    bc += 1f
    for(i<-0 until OC ; j<-0 until IC ; k<-0 until KW ; l<-0 until KW){
      for(m<-0 until OH ; n<-0 until OW){
        dK(i)(j)(k)(l) += G(i*OH*OW + m*OW+n) * Vtemp(j*IH*IW + (m+k)*IW+(n+l))
      }
    }
    var dV = Array.ofDim[T](IC * IH * IW)
    for(i<-0 until IC ; j<-0 until IH ; k<-0 until IW){
      for(l<-0 until OH ; m<-0 until KW){
        if(l + m == j){
          for(n<-0 until OW ; p<-0 until KW){
            if(n + p == k){
              for(q<-0 until OC){
                dV(i*IW*IH + j*IW+k) += K(q)(i)(m)(p)  * G(q*OH*OW + l*OW+n)
              }
            }
          }
        }
      }
    }
    dV
  }


 override def backward(G:Array[Array[T]])={
    G.reverse.map(backward).reverse
  }

/*  def backward(ds:Array[Array[Float]]) = {
    val dloss = ds.map(_.transpose(OC, OH*OW)).flatten

    // backward kernel
    val past_xs = xs.take(ds.size).reverse.toArray.flatten
    xs = xs.drop(ds.size)
    val dkernels = Array.ofDim[Float](KW*KW*IC * OC)
    blas.BLAS.matmulF(past_xs.transpose(OH*OW*ds.size, KW*KW*IC), dloss, dkernels, KW*KW*IC, OC, OH*OW*ds.size)
    Range(0, grad.size).foreach(i => grad(i) += dkernels(i))

    // backward input

    val dinput = Array.ofDim[Float](OH*OW*ds.size*KW*KW*IC)
    blas.BLAS.matmulF(dloss, K.flatten.flatten.flatten.transpose(KW*KW*IC, OC), dinput, OH*OW*ds.size, KW*KW*IC, OC)

    dinput.grouped(OH*OW * KW*KW*IC).map(this.reconvert).toArray
  }*/


  def update() {
    val p2 = 0.999f //モーメントの推定に対する指数減衰率
    val delta = 0.00000001f //数値安定のための小さな定数デルタ
    p1t = p1t * p1 //updata（p1）が何回呼び出されたか
    p2t = p2t * p2
    for(i<-0 until OC ; j<-0 until IC ; k<-0 until KW ; l<-0 until KW){
      s(i)(j)(k)(l) = p1 * s(i)(j)(k)(l) + (1f-p1) * dK(i)(j)(k)(l)
      r(i)(j)(k)(l) = p2 * r(i)(j)(k)(l) + (1f-p2) * ( dK(i)(j)(k)(l) * dK(i)(j)(k)(l) )
      val s_ = s(i)(j)(k)(l) / (1f-p1t)
      val r_ = r(i)(j)(k)(l) / (1f-p2t)
      K(i)(j)(k)(l) = K(i)(j)(k)(l) - ep * s_ / ( math.sqrt(r_).toFloat + delta )
    }
    reset()
  }

  override def save(fn:String){
    val save = new java.io.PrintWriter(fn)
    for(i<-0 until OC ; j<-0 until IC ; k<-0 until KW ; l<-0 until KW){
      save.println(K(i)(j)(k)(l))
    }
    save.close
  }
  override def load(fn:String){
    val f = scala.io.Source.fromFile(fn).getLines.toArray
    for(i<-0 until OC ; j<-0 until IC ; k<-0 until KW ; l<-0 until KW){
      K(i)(j)(k)(l) = f(i*IC*KW*KW + j*KW*KW + k*KW + l).toFloat
    }
  }
  def reset() {
    dK = Array.ofDim[T](OC,IC,KW,KW)
  }
  def convert(x:Array[Float]) = {
    val f_height = OH * OW
    val f_width = KW * KW * IC
    val features = Array.ofDim[Float](f_height*f_width)

    def findex(i:Int, j:Int, k:Int, l:Int, m:Int) =
      i*f_width*OW+j*f_width+k*KW*KW+l*KW+m

    def xindex(i:Int, j:Int, k:Int, l:Int, m:Int) =
      i*IW+j+k*IW*IH+l*IW+m

    for(
      i <- 0 until OH;
      j <- 0 until OW;
      k <- 0 until IC;
      l <- 0 until KW;
      m <- 0 until KW
    ) features(findex(i,j,k,l,m)) = x(xindex(i,j,k,l,m))
    features
  }

  def reconvert(x:Array[Float]) = {
    val f_height = OH * OW
    val f_width = KW * KW * IC
    val features = Array.ofDim[Float](IH*IW*IC)

    def findex(i:Int, j:Int, k:Int, l:Int, m:Int) =
      i*f_width*OW+j*f_width+k*KW*KW+l*KW+m

    def xindex(i:Int, j:Int, k:Int, l:Int, m:Int) =
      i*IW+j+k*IW*IH+l*IW+m

    for(
      i <- 0 until OH;
      j <- 0 until OW;
      k <- 0 until IC;
      l <- 0 until KW;
      m <- 0 until KW
    ) features(xindex(i,j,k,l,m)) += x(findex(i,j,k,l,m))

    features
  }


  override def add_b(Z:Array[T] , fn:String)={
    val f = scala.io.Source.fromFile(fn).getLines.toArray
    for(i<-0 until OC ; j<-0 until OH ; k<-0 until OW){
      for(l<-0 until IC ; m<-0 until KW ; n<-0 until KW){
        Z(i*OH*OW + j*OW+k) += f(i).toFloat
      }
    }
    Z
  }
}
