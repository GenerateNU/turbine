package driver 

import org.apache.spark.sql.{Dataset, DataFrame}
import org.apache.spark.sql.functions.col

object FileFilter {
  def filterFilesByExtensions(repoFiles: Dataset[PreprocessGitHubRepos.RepoFile], extensions: Seq[String]): Dataset[PreprocessGitHubRepos.RepoFile] = {
    repoFiles.filter(file => extensions.exists(file.fileName.endsWith))
  }
}
