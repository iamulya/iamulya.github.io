---
layout: postwithads
title: Setup a Cloud Workstations Playground using Terraform
date: 2024-08-08 09:00:00 +0100
categories: [Google Cloud, Cloud Workstations]
tags: [Google Cloud, Cloud Workstations, DX, Developer Experience, CDE, Terraform]
---

I wrote recently on the official [Google Cloud blog](https://cloud.google.com/blog/products/application-modernization/dz-bank-improves-developer-productivity-with-cloud-workstations) about how I helped one of Germany's biggest banks improve their Developer Experience using Cloud Workstations. 

Cloud Workstations are preconfigured development environments hosted in the cloud. They provide developers with a consistent, scalable, and secure workspace to build and test applications. Think of them as virtual machines (VMs) tailored specifically for development, with additional features and benefits. 

### Key Features and Benefits

* **On-demand access**: Start and stop workstations as needed, saving costs.   
* **Scalability**: Adjust resources (CPU, RAM, storage) based on project requirements.   
* **Security**: Benefit from Google Cloud's robust security measures to protect your code and data.   
* **Collaboration**: Share workstations with team members for efficient development.   
* **Integration**: Seamlessly integrate with other Google Cloud services like Cloud Storage, BigQuery, and Kubernetes.
* **Customization**: Create custom workstations with specific tools and configurations.

For those who want to take home these advantages, I have created a terraform script to help you get started. You can find the code [here](https://github.com/iamulya/cloud-workstation-tf-setup). This will provide you with all the necessary setup (Network, Cluster, Config etc.) to start and use your Workstation. The script runs for around 20-25 minutes to create all the resources and outputs two Workstations: one with VSCode and the other with Intellij, as the primary IDE.

The VS Code Workstation can directly be launched in the browser, whereas in order to launch the Intellij Workstation, you'd need to use [JetBrains Gateway](https://www.jetbrains.com/remote-development/gateway/). The whole process is defined [here](https://cloud.google.com/workstations/docs/develop-code-using-local-jetbrains-ides). 

### Next Steps

[Customize your workstations specific to your own requirements](https://cloud.google.com/workstations/docs/customize-container-images)

[Connect to your workstation using SSH](https://cloud.google.com/workstations/docs/ssh-support)
