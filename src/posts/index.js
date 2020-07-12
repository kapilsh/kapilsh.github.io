import React from "react";
import setupImage from "../../static/linux.jpg";
import ridgeLassoImage from "../../static/ridge_lasso.png";
import localRegressionSmootherImage from "../../static/LocalLinearSmoother.png";
import kernelSmootherImage from "../../static/KernelSmoother.png";
import networkImage from "../../static/network.jpg";
import matrixImage from "../../static/matrix.png";
import linRegImage from "../../static/lin_reg.jpg";
import regressionImage from "../../static/regression.png";

import MyPersonalSetup from "../components/blog/contents/MyPersonalSetup";
import RegularizationInLinearModels from "../components/blog/contents/RegularizationInLinearModels";
import LocalLinearRegression from "../components/blog/contents/LocalLinearRegression";
import KernelSmoothing from "../components/blog/contents/KernelSmoothing";
import ZeroMQProxy from "../components/blog/contents/ZeroMQProxy";
import EigenBenchmarks from "../components/blog/contents/EigenBenchmarks";
import GramSchmidtProcedure from "../components/blog/contents/GramSchmidtProcedure";
import GeometryofRegression from "../components/blog/contents/GeometryofRegression";

const posts = [
  {
    title: "My Personal Setup",
    tags: ["Linux", "General"],
    description: "Setup of my Linux Development Environment",
    image: setupImage,
    date: "January 12, 2019",
    content: `This is more of a personal to-do list to setup a new linux development
    environment. Hope, this helps others to setup their own.`,
    component: <MyPersonalSetup />,
  },
  {
    title: "Regularization in Linear Models",
    tags: ["Python", "Machine Learning", "Math"],
    description: "Ridge and Lasso Regression",
    image: ridgeLassoImage,
    date: "September 14, 2018",
    content: `Least squares estimates are often not very satisfactory due to their
    poor out-of-sample performance, especially when the model is overly
    complex with a lot of features. Regularization is a method to shrink or
    drop coefficients/parameters from a model by imposing a penalty on their
    size. It is also referred to as the Shrinkage Method. In this post, I
    will discuss two of the most common regularization techniques - Ridge
    and Lasso regularization.`,
    component: <RegularizationInLinearModels />,
  },
  {
    title: "Local Linear Regression",
    tags: ["Python", "Machine Learning", "Math"],
    description: "Moving from Locally Weighted Constants to Lines",
    image: localRegressionSmootherImage,
    date: "August 31, 2018",
    content:
      "I previously wrote a post about Kernel Smoothing and how it can be used to fit a non-linear function non-parametrically. In this post, I will extend on that idea and try to mitigate the disadvantages of kernel smoothing using Local Linear Regression.",
    component: <LocalLinearRegression />,
  },
  {
    title: "Kernel Smoothing",
    tags: ["Python", "Machine Learning", "Math"],
    description: "Gaussian Kernel Smoothing and Optimal Bandwidth Selection",
    image: kernelSmootherImage,
    date: "August 26, 2018",
    content:
      "Kernel Method is one of the most popular non-parametric methods to estimate probability density and regression functions. As the word Non-Parametric implies, it uses the structural information in the existing data to estimate response variable for out-of-sample data.",
    component: <KernelSmoothing />,
  },
  {
    title: "ZeroMQ Proxy",
    tags: ["Java", "ZeroMQ", "Microservices"],
    description: "How to Solve the Dynamic Discovery Problem in ZeroMQ",
    image: networkImage,
    date: "May 16, 2018",
    content:
      "ZeroMQ is my favorite message passing and networking library. It has bindings for almost all major languages and it’s super convenient to build polyglot distributed network applications with it. Also, ZeroMQ documentation and examples are very exhaustive.",
    component: <ZeroMQProxy />,
  },
  {
    title: "Eigen Benchmarks",
    tags: ["C++", "Eigen", "SpdLog", "Conan"],
    description: "Matrix Inversion using Eigen C++ Library",
    image: matrixImage,
    date: "January 28, 2018",
    content:
      "Eigen is super fast linear algebra library for C++. It provides almost all matrix / vector related operations and some extra pandas / numpy style functionality. Recently, one of my colleagues was looking for a linear algebra for C++ and I suggested using Eigen. During our conversation, we were discussing how fast are matrix inverse operation in Eigen, however the Eigen docs did not provide a satisfactory benchmarks for inversion. So, I decided to do a little test on my own.",
    component: <EigenBenchmarks />,
  },
  {
    title: "Gram Schmidt Procedure",
    tags: ["Python", "Machine Learning", "Math"],
    description: "Solving Regression using Gram-Schmidt Procedure",
    image: linRegImage,
    date: "January 07, 2018",
    content:
      "An interesting way to understand Linear Regression is Gram-Schmidt Method of successive projections to calculate the coefficients of regression. Gram-Schmidt procedure transforms the variables into a new set of orthogonal or uncorrelated variables. On applying the procedure, we should get exactly the same regression coefficients as with projection of predicted variable on the feature space.",
    component: <GramSchmidtProcedure />,
  },
  {
    title: "Geometry of Regression",
    tags: ["Python", "Machine Learning"],
    description: "Geometric Interpretation of Linear Regression",
    image: regressionImage,
    date: "March 05, 2017",
    content: "A picture is worth a thousand words. ",
    component: <GeometryofRegression />,
  },
];

export default posts;