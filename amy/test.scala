object HelloWorld {
        def main(args:Array[String]){
                val path = "file:///Users/chaijun/Desktop/tx/fmTest/kb_click/";
val rdd1 = sc.textFile(path,2); 
val array=rdd1.collect() ;
val result=for (x <-array) yield(x.split("\\s+")(0)+","+x.split("\\s+")(1));
val ss=result.distinct;
println(ss.toString);
        }
}
