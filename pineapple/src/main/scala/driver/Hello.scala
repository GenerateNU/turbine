package driver

import sys.process._
import io.circe.parser

object GitHubRepoDownloader {
  def main(args: Array[String]): Unit = {
    // error handling with arg parsing
    if (args.length != 1) {
      println("Usage: GitHubRepoDownloader <GitHub_API_URL>")
      System.exit(1)
    }

    val apiUrl = args(0)
    val githubAccessToken = "PUT AT"

    // fetch user's repos using a get request
    val repoInfo = scala.io.Source.fromURL(apiUrl, headers = Seq("Authorization" -> s"token $githubAccessToken")).mkString

    // parse JSON response
    // get name and url 
    val repositories = io.circe.parser.parse(repoInfo) match {
      case Right(json) =>
        json.asArray.getOrElse(Vector.empty).map { repo =>
          (repo.hcursor.downField("name").as[String].getOrElse(""), repo.hcursor.downField("clone_url").as[String].getOrElse(""))
        }
      case Left(error) =>
        println(s"Failed to parse JSON: $error")
        Vector.empty
    }

    // clone repos
    repositories.foreach { case (repoName, cloneUrl) =>
      println(s"Cloning $repoName...")
      s"git clone $cloneUrl".!
    }
  }
}
