package driver


import org.apache.spark.sql.{SparkSession, Dataset}
import org.apache.spark.sql.functions._

object ProcessRepos {
  case class RepoFile(repoName: String, fileName: String, fileContent: String)

  def main(args: Array[String]): Unit = {
    if (args.length != 1) {
      println("Usage: PreprocessGitHubRepos <path_to_cloned_repositories>")
      System.exit(1)
    }

    val clonedReposPath = args(0)

    val spark = SparkSession.builder()
      .appName("PreprocessGitHubRepos")
      .getOrCreate()

    import spark.implicits._

    // Read all files from cloned repositories
    val repoFiles: Dataset[RepoFile] = spark.read
      .textFile(s"$clonedReposPath/*/*")
      .flatMap { line =>
        val lines = line.split("\n")
        val repoName = lines.headOption.getOrElse("")
        val fileName = lines.lift(1).getOrElse("")
        val fileContent = lines.drop(2).mkString("\n")
        if (repoName.nonEmpty && fileName.nonEmpty && fileContent.nonEmpty)
          Some(RepoFile(repoName, fileName, fileContent))
        else
          None
      }

    // Define the list of file extensions to filter by
    val allowedExtensions = Seq(".go", ".tsx", ".py", ".js", ".css", ".yaml", ".yml")

    // Use the FileFilter object to filter files by extensions
    val filteredFiles = FileFilter.filterFilesByExtensions(repoFiles, allowedExtensions)


    // Show the first few records as a demonstration
    filteredFiles.show()

    // Stop the Spark session when done
    spark.stop()
  }
}
