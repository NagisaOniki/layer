// package layer
// import layer.Adam._
// import breeze.linalg._
// import pll.utils.RichArray._
// class Convolution2D(
//   val kw:Int,
//   val H:Int, //入力の高さ
//   val W:Int, //入力の幅
//   val I:Int, //入力チャネル数
//   val O:Int, //出力チャネル数
//   val eps:Float=0.001f
// ) extends Layer {
//   override type T = Float
//   var X = Array[T]()
//   val rand = new scala.util.Random(0)
//   var K=Array.ofDim[T](O*I*kw*kw).map(a => rand.nextGaussian.toFloat *0.01f)
//   var t=0f
//   val w_d=W-kw+1
//   val h_d=H-kw+1
//   val h = kw*kw*I
//   val w = (W-kw+1)*(H-kw+1)//OW*OH
//   var d_k=Array.ofDim[T](O*h)
//   override def load(fn:String) {
//     val f =scala.io.Source.fromFile(fn).getLines.map(_.split(",").map(_.toFloat).toArray).toArray
//     for(i <- 0 until O*I*kw*kw){
//       K(i)=f(0)(i)
//     }
//   }
//   override def save(fn:String) {
//     val file=new java.io.PrintWriter(fn)
//     for(i <- 0 until O*I*kw*kw){
//       if(i == O*I*kw*kw -1){file.print(K(i))}
//       else{
//         file.print(K(i)+",")
//       }
//     }
//     file.close
//   }
//   def transpose(x:Array[T],row:Int,col:Int)={
//     var xt = Array.ofDim[T](row*col)
//     for(i <- 0 until col;j <- 0 until row){
//       xt(i*row+j) = x(i+col*j)
//     }
//     xt
//   }
//   def vind(i:Int,j:Int,k:Int)=i*H*W+j*W+k
//   def zind(i:Int,j:Int,k:Int)=i*w+j*(W-kw+1)+k
//   def kind(i:Int,j:Int,k:Int,l:Int)=i*h+j*kw*kw+k*kw+l
//   val OW = W-kw+1
//   val OH = H-kw+1
//   override def forward(xs:Array[Array[T]]):Array[Array[T]] ={
//     X = xs.flatten
//     var x_d = Array.ofDim[T](h*xs.size,w)
//     var ys1 = Array.ofDim[T](O*xs.size*w)
//     var ys = Array.ofDim[T](O*xs.size*w)
//     var tmp = Array.ofDim[T](h*xs.size,w)
//     var row = 0
//     for(n <- 0 until xs.size){
//       for(i <- 0 until I;j <- 0 until H-kw+1;k <- 0 until W-kw+1){
//         for(l <- 0 until kw;m <- 0 until kw){
//           xs(n)(i*H*W+j*W+k+l*W+m)
//         }
//         row+=1
//       }
//     }
//     var index=0
//     for(i <- 0 until h;j <- 0 until xs.size){
//       x_d(index)=tmp(j*h+i)
//       index+=1
//     }
//     blas.BLAS.matmulF(K,x_d.flatten,ys1,O,w*xs.size,h)
//     X = x_d.flatten
//     index=0
//     for(i <- 0 until xs.size;j <- 0 until O;k <- 0 until w){
//       ys(index)=ys1(i*w+j*xs.size*w+k)
//       index+=1
//     }
//     ys.grouped(w*O).toArray
//   }
//   def forward(x:Array[T])={
//     var x_d=Array.ofDim[T](h,w)
//     var row = 0
//     var y = Array.ofDim[T](w*O)
//     for(i <- 0 until I;j <- 0 until H-kw+1;k <- 0 until W-kw+1){
//       for(l <- 0 until kw;m <- 0 until kw){
//         x_d(row)(l*kw+m) = x(i*H*W+j*W+k+l*W+m)
//       }
//       row+=1
//     }
//     X = x_d.flatten
//     blas.BLAS.matmulF(K,x_d.flatten,y,O,w,h)
//     y
//   }
//   var count = 0f
//   override def backward(Gs:Array[Array[T]]):Array[Array[T]]={
//     count += 1f
//     val n = Gs.size
//     var Gd = Gs.flatten
//     var ds = Array.ofDim[T](O*n*w)
//     var dk = Array.ofDim[T](O*h)
//     var dv = Array.ofDim[T](h*n*w)
//     var d_v = Array.ofDim[T](n,I*H*W)
//     var index=0
//     val xt  = transpose(X,h,w*n)
//     for(i <- 0 until O;j <- 0 until n;k <- 0 until w){
//       ds(index)=Gd(i*w+j*w*O+k)
//       index+=1
//     }
//     blas.BLAS.matmulF(ds,transpose(X,h,w*n),dk,O,h,w*n)
//     blas.BLAS.matmulF(transpose(K,O,h),ds,dv,h,w*n,O)
//     var tmp = Array.ofDim[T](n,h*w)
//     for(i <- 0 until n;j <- 0 until h;k <- 0 until w){
//       tmp(i)(j*w+k) = dv(j*w*n+i*w+k)
//     }
//     for(p <- 0 until n){
//       index = 0
//       for(i<-0 until I ; j<-0 until H-kw+1 ; k<-0 until W-kw+1 ; l<-0 until kw ; m<-0 until kw){
//         d_v(p)(i*H*W+j*W+k+l*W+m) += tmp(p)(index)
//         index+=1
//       }
//     }
//     d_k = dk.grouped(O*h).toArray.flatten
//     d_v
//   }
//   def backward(G:Array[T])={
//     count += 1f
//     var dv = Array.ofDim[T](h*w)
//     var d_v = Array.ofDim[T](I*H*W)
//     blas.BLAS.matmulF(G,transpose(X,h,w),d_k,O,h,w)
//     blas.BLAS.matmulF(transpose(K,O,h),G,dv,h,w,O)
//     var index = 0
//     for(i <- 0 until I;j <- 0 until H-kw+1;k <- 0 until W-kw+1;l <- 0 until kw;m <- 0 until kw){
//       d_v(i*H*W+j*W+k+l*W+m) += dv(index)
//       index+=1
//     }
//     d_v
//   }
//   var rt1=1f
//   var rt2=1f
//   var s=Array.ofDim[T](O*I*kw*kw)
//   var r=Array.ofDim[T](O*I*kw*kw)
//   var adam_k = new Adam_D(K.size,eps)
//   def update()={
//     adam_k.update(K,d_k,count)
//     reset()
//   }
//   def update2(){
//     var lr=0.001f
//     for(i <- 0 until K.size ){
//       K(i) -= lr * d_k(i)
//     }
//     reset()
//   }
//   def reset() {
//     d_k=Array.ofDim[T](O*I*kw*kw)
//   }
// }
