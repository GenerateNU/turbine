import Dependencies._

ThisBuild / scalaVersion     := "2.13.11"
ThisBuild / version          := "0.1.0-SNAPSHOT"
ThisBuild / organization     := "com.example"
ThisBuild / organizationName := "example"

lazy val root = (project in file("."))
  .settings(
    name := "pineapple",
    libraryDependencies += munit % Test
  )

libraryDependencies += "org.apache.httpcomponents" % "httpclient" % "4.5.13"
libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % "3.0.3",
  "org.apache.spark" %% "spark-sql" % "3.0.3"
)

libraryDependencies ++= Seq(
  "io.circe" %% "circe-core" % "0.14.1",        // Core Circe functionality
  "io.circe" %% "circe-generic" % "0.14.1",    // For automatic derivation of codecs
  "io.circe" %% "circe-parser" % "0.14.1"      // For parsing JSON
)
// See https://www.scala-sbt.org/1.x/docs/Using-Sonatype.html for instructions on how to publish to Sonatype.
