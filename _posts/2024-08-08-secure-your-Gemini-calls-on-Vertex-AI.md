---
layout: postwithads
title: Secure your Gemini Calls on Vertex AI
date: 2024-08-08 12:00:00 +0100
categories: [Google Cloud, Vertex AI, Gemini]
tags: [Google Cloud, Vertex AI, Gemini, VPC Service Controls, Access Context Manager]
---

As Gen AI gets more popular and integrated into corporate strategies, the use of models like Gemini, GPT 4, Llama etc. is increasing in an incredibly fast fashion. However, a lot of the times the security aspect is put on the backburner - which is a big mistake. 

In this article, I'll share with you how to secure your calls to Gemini on Vertex AI, making sure only certain IPs or identities are able to use the service. In order to achieve this, we'll be using two Google Cloud services:

#### Access Context Manager

Access Context Manager (ACM) is a security tool in Google Cloud that enables you to implement fine-grained, attribute-based access control for your cloud resources. It provides a powerful mechanism to define and enforce security policies based on various contextual factors.

How it Works:

* **Access Policies**: These are organization-wide containers that hold access levels and service perimeters.
* **Access Levels**: These define the conditions that a request must meet to be granted access. You can specify requirements based on:
1. Device type and operating system
2. IP address
3. User identity
4. Geographic location and other attributes
* **Service Perimeters**: These define the boundaries around your cloud resources, specifying which projects and services are included.

#### VPC Service Controls

VPC Service Controls is a security feature in Google Cloud that enables you to create perimeters around your cloud resources to protect them from unauthorized access. It's an additional layer of security that complements Identity and Access Management (IAM).

How it works:

* **Service Perimeters**: You define boundaries around your cloud resources, specifying which projects and services are included.

* **Access Control**: By default, resources within a perimeter are inaccessible from outside the perimeter. This helps prevent data exfiltration and unauthorized access.

* **Ingress and Egress Rules**: You can define rules to allow specific access to resources within or outside the perimeter, providing granular control over data flow.

* **Context-Aware Access**: VPC Service Controls can consider attributes like user identity, IP address, and location to make access decisions, enhancing security.

#### Create an access policy

What we need an access policy (a container for access level and service perimeter) for our purposes. We'll use Access Context Manager to create an access level defining the IP Address or the CIDR Range which will be allowed to access the service. 

![ ](/assets/img/access-cm.jpg)

Here you can also restrict location, device and other parameters. 

Next, we move to VPC Service Controls to define a service perimeter using the access level and other attributes. Make sure you first create the perimeter in dry run mode before trying the enforced mode.

![ ](/assets/img/perimeter.jpg)

You can select which identities (Users/Service Accounts) are allowed make the calls and also if you want to restrict specific projects.

This setup will give you a more secure access to your Vertex AI APIs, since you can now restrict access based on IP/User/Service Account/Region etc.
