---
title: A Cheatsheet to Level Up Your Terraform
date: 2025-06-27 14:00:00 +0100
categories: [Cloud-Native, IaC, Terraform]
tags: [Cloud-Native, IaC, Terraform, Infrastructure as Code, Cheatsheet]
image:
  path: /assets/img/tf-post.png
  alt: "TF Cheatsheet"
---

So you've mastered the basics. You can write a resource, run terraform plan, and apply with confidence. Your infrastructure is code, but you're starting to feel the growing pains. Your configurations are getting repetitive, refactoring is scary, and managing multiple environments feels clunky.

It’s time to level up.

This guide moves beyond the fundamentals and into the patterns, functions, and architectural decisions that separate a simple configuration from a scalable, maintainable, and production-grade Infrastructure as Code (IaC) system. We'll cover how to write smarter HCL and automate with confidence.

### Mastering HCL: From Static to Dynamic Code

The key to reducing repetition and increasing flexibility is to master Terraform's HashiCorp Configuration Language (HCL). This means moving from static resource blocks to dynamic, data-driven configurations.

#### Embrace `for_each`, Abandon `count`

This is the most important lesson for intermediate users. While `count` seems intuitive for creating multiple resources, it's dangerously fragile. If you have a list of three servers created with `count` and you remove the second one, Terraform sees the list shrink and re-numbers the third server, causing it to be destroyed and recreated.

`for_each` solves this by iterating over a map or a set, using the key or value as a persistent identifier in the state file.

```hcl
# BAD: Fragile and causes unnecessary changes.
variable "subnets_list" { 
    type    = list(string), 
    default = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"] 
}

resource "aws_subnet" "bad" {
  count      = length(var.subnets_list)
  cidr_block = var.subnets_list[count.index]
  # ...
}

# GOOD: Resilient. Removing a CIDR block from the set has no impact on the others.
variable "subnets_set" { 
    type = set(string), 
    default = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
}

resource "aws_subnet" "good" {
  for_each   = var.subnets_set
  cidr_block = each.value # each.key and each.value are the same for a set
  # ...
}
```

#### Generate Configuration with `dynamic` Blocks

Ever needed to create a variable number of security group rules or ALB listener rules? `dynamic` blocks are your answer. They let you programmatically generate nested blocks inside a resource.

```hcl
variable "ingress_ports" {
  description = "A list of ports to allow ingress."
  type        = list(number)
  default     = [80, 443]
}

resource "aws_security_group" "web" {
  name = "web-server-sg"

  # For each port in the list, create an ingress block
  dynamic "ingress" {
    for_each = toset(var.ingress_ports)
    content {
      from_port   = ingress.value
      to_port     = ingress.value
      protocol    = "tcp"
      cidr_blocks = ["0.0.0.0/0"]
    }
  }
}
```

#### Assertions and Guarantees

- **`precondition` Blocks:** Validate input variables *within the context of a resource*. This is more powerful than a `variable` validation block because it can use other data and resources.
    
```hcl
resource "aws_instance" "database" {
    # ...
    ami = data.aws_ami.db_ami.id

    lifecycle {
    # Fails the plan if the selected AMI is not tagged as 'DBCertified'.
    precondition {
        condition     = can(data.aws_ami.db_ami.tags["DBCertified"])
        error_message = "The selected database AMI must be certified. Check its tags."
    }
    }
}

```
    
- **`postcondition` Blocks:** Validate the outcome of a resource *after* it has been created or changed. Useful for checking properties that are only known after creation.

```hcl
resource "aws_instance" "web" {
    # ...

    lifecycle {
    postcondition {
        condition     = self.public_ip == null
        error_message = "The web server must not have a public IP address."
    }
    }
}
```
#### Transform Data with `for` Expressions and `templatefile`

Your code shouldn't just declare resources; it should prepare and transform data.
*   **`for` expressions** create new lists or maps from existing ones, perfect for restructuring data before feeding it to a resource.

```hcl
# Use case: Create a flat list of firewall rules.
variable "app_firewall_rules" {
  type = map(object({
    ports    = list(number)
    protocol = string
  }))
  default = {
    "http"  = { ports = [80, 8080], protocol = "tcp" },
    "https" = { ports = [443], protocol = "tcp" }
  }
}

locals {
  # Creates a flat list: [{port=80, proto="tcp"}, {port=8080, proto="tcp"}, {port=443, proto="tcp"}]
  flattened_rules = flatten([
    for app_key, app_config in var.app_firewall_rules : [
      for port in app_config.ports : {
        port  = port
        proto = app_config.protocol
      }
    ]
  ])
}
```
*   **`templatefile`** is the modern, safe way to generate configuration files like `cloud-init` scripts, populating them with variables from your Terraform code.

```hcl
# Use 'for' to transform a simple list into a structured map for for_each
variable "user_names" { default = ["alice", "bob"] }

locals {
  iam_users = {
    for name in var.user_names : name => {
      arn  = "arn:aws:iam::123456789012:user/${name}"
      path = "/system/"
    }
  }
}

# Use templatefile to create a dynamic user_data script
resource "aws_instance" "web" {
  # ...
  user_data = templatefile("${path.module}/templates/init.sh.tftpl", {
    hostname   = "web-server-01"
    admin_user = "deployer"
  })
}
```

### State Management and Refactoring Like a Pro

Refactoring used to involve risky `terraform state mv` commands. Modern Terraform provides declarative, version-controlled ways to manage state changes safely.

*   **`moved` Blocks:** When you rename a resource or move it into a module, add a `moved` block. This tells Terraform it's the same resource at a new address, preventing a destructive plan.

    ```hcl
    # To refactor aws_instance.server to aws_instance.web_server, add this block:
    moved {
      from = aws_instance.server
      to   = aws_instance.web_server
    }
    ```

*   **`import` Blocks:** To bring existing, manually-created infrastructure under Terraform's management, define the resource in your code and add an `import` block. It’s declarative and part of the plan, making it far safer than the old CLI command.

    ```hcl
    import {
      to = aws_s3_bucket.legacy_app
      id = "my-legacy-app-bucket" # The ID the provider uses to find the resource.
    }
    ```

After applying the change, the `moved` or `import` block can be safely removed.

### Bulletproof Automation and CI/CD

#### Policy as Code (PaC)

Integrate policy enforcement into your pipeline to prevent deployment of non-compliant infrastructure.

- **HashiCorp Sentinel:** Integrated with Terraform Cloud/Enterprise. Enforces policy between plan and apply.
- **Open Policy Agent (OPA):** Open-source standard. Use with tools like `conftest` to test your Terraform plan JSON against policies.

**Example Use Case for PaC:** "Disallow any security group that has an ingress rule open to `0.0.0.0/0` on port `22`."

#### Secrets Management Integration

Never store secrets in `.tfvars` or Git. Use a dedicated provider to fetch them dynamically at plan/apply time.

```hcl
# Use the Vault provider to fetch a secret
provider "vault" {
  address = "<https://vault.example.com>"
}

data "vault_generic_secret" "db_creds" {
  path = "secret/data/database/credentials"
}

resource "aws_db_instance" "main" {
  # The sensitive value is only ever in memory during the Terraform run
  username = data.vault_generic_secret.db_creds.data["username"]
  password = data.vault_generic_secret.db_creds.data["password"]
}

```

#### `terraform_remote_state` Data Source

This is the glue that connects separate Terraform configurations. It allows one configuration to read the `output` values of another.

```hcl
data "terraform_remote_state" "network" {
  backend = "s3"
  config = {
    bucket = "my-terraform-state-bucket"
    key    = "network/prod/terraform.tfstate"
    region = "us-east-1"
  }
}

resource "aws_instance" "app_server" {
  # Use the VPC ID and subnet ID from the network state
  vpc_id     = data.terraform_remote_state.network.outputs.vpc_id
  subnet_id  = data.terraform_remote_state.network.outputs.private_subnet_ids[0]
}
```

#### The Ideal CI/CD Pipeline Flow

1. **On Pull Request:**
    - `terraform init`
    - `terraform validate`
    - `terraform fmt --check` (Fail if not formatted)
    - **Static Code Analysis:** `tfsec` or `checkov` to find security issues.
    - **Policy Check (OPA):** `conftest test plan.json` to enforce custom rules.
    - `terraform plan -out=tfplan` (Save the plan artifact)
    - **Post Plan Output:** Use a tool like `terraform-plan-parser` to post a human-readable summary as a PR comment.
2. **On Merge to `main`/`production` Branch:**
    - **Manual Approval Gate:** A required manual click in the CI/CD system (e.g., GitHub Environments, GitLab Protected Environments) before applying. The approver should review the saved plan.
    - `terraform apply "tfplan"`

#### Automated Variable Management

Avoid manually managing `.tfvars` files across many environments. Instead, generate them from a central source of truth.

**Pattern:** Use a YAML file as your source of truth and a simple script to generate `terraform.tfvars.json` for each environment.

```yaml
# config/environments.yaml
global:
  region: "us-east-1"
environments:
  dev:
    instance_type: "t3.micro"
    instance_count: 1
  prod:
    instance_type: "m5.large"
    instance_count: 10

```

A CI script can then parse this YAML and generate the correct `terraform.tfvars.json` before running `plan`.

### The "Escape Hatch": Provisioners & Null Resources

Provisioners should be a **last resort**. Always prefer declarative solutions. But when you need them, use them correctly.

| Tool | Description & Use Case |
| --- | --- |
| **`local-exec` Provisioner** | Runs a command on the machine where you run Terraform. <br/>**Use Case:** Running a script to perform an action on an API that has no Terraform provider. |
| **`remote-exec` Provisioner** | Runs a command on the created resource over SSH/WinRM. <br/>**Use Case:** (Legacy) Running a configuration management script. **Strongly prefer `user_data` or dedicated CM tools instead.** |
| **`null_resource`** | A resource that does nothing. Its purpose is to act as a container for provisioners or as a trigger for other actions. <br/>**Use Case:** Run a provisioner only when a specific value changes (e.g., an S3 object is updated). |

```hcl
# Use Case: Run a database migration script only when the app version changes.
resource "null_resource" "db_migrator" {
  # The 'triggers' map forces this resource to be replaced if any of its values change.
  triggers = {
    app_version = var.application_version
    db_address  = aws_db_instance.main.address
  }

  provisioner "local-exec" {
    # This script runs on replacement, i.e., when the app_version changes.
    command = "db-migrate --host=${self.triggers.db_address} --version=${self.triggers.app_version}"
  }
}

```

**⚠️ When to Avoid Provisioners:**

- **For VM Image Creation:** Use Packer.
- **For System Configuration:** Use `cloud-init` (`user_data`), Ansible, Puppet, or Chef.
- **For Application Deployment:** Use container orchestrators (ECS, Kubernetes), Lambda, or dedicated deployment tools.

### Performance & CLI Power-Ups

| Command / Flag | Description & Best Practice |
| --- | --- |
| `terraform plan -refresh=false` | Skips synchronizing the state file with real-world infrastructure. <br/>**Use Case:** Massively speeds up planning in very large configurations where you know no out-of-band changes have occurred. |
| `terraform plan -target=...` | Generates a plan for only the specified resource and its dependencies. <br/>**Warning:** **USE WITH EXTREME CAUTION.** This can cause state drift. It's a debugging tool, not a workflow. |
| `terraform console` | An interactive console (REPL) for experimenting with expressions, functions, and variable values without running a full plan. <br/>Invaluable for debugging complex logic. |
| `terraform test` | (Terraform 1.6+) A built-in framework for writing formal, repeatable tests for your modules, including creating and destroying real infrastructure. <br/>A huge step up from manual testing. |
| `TF_LOG=TRACE`	| The ultimate debugging tool. Sets the log level for the Terraform run. Levels: TRACE, DEBUG, INFO, WARN, ERROR. <br/> Use Case: TF_LOG=TRACE terraform plan will show you detailed API calls, provider logic, and internal graph processing. <br/>Essential for diagnosing provider bugs or complex issues. | 
| `terraform graph` | Outputs the dependency graph in DOT format. <br/> Use Case: Pipe the output to dot (from GraphViz) to generate an image of your resource dependency graph. <br/>`terraform graph | dot -Tpng > graph.png` gives you a visual representation of your infrastructure, which is invaluable for understanding complex configurations. | 

### Final Words

I hope this guide has given you the tools and patterns to take your Terraform skills to the next level. Remember, IaC is not just about writing code; it's about creating a maintainable, scalable, and resilient infrastructure that can adapt to your needs. Happy building!