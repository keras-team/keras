# Security Policy

 - [**Using Keras Securely**](#using-keras-securely)
   - [Untrusted inputs](#untrusted-inputs)
   - [Data privacy](#data-privacy)
   - [Untrusted environments or networks](#untrusted-environments-or-networks)
   - [Multi-Tenant environments](#multi-tenant-environments)
 - [**Reporting a Vulnerability**](#reporting-a-vulnerability)

## Using Keras Securely

### Untrusted inputs

Some models accept various input formats (text, images, audio, etc.). The libraries converting these inputs have varying security levels, so it's crucial to isolate the model and carefully pre-process inputs to mitigate script injection risks.

For maximum security when handling untrusted inputs, you may need to employ the following:

* Sandboxing: Isolate the model process.
* Pre-analysis: check how the model performs by default when exposed to prompt injection (e.g. using [fuzzing for prompt injection](https://github.com/FonduAI/awesome-prompt-injection?tab=readme-ov-file#tools)). This will give you leads on how hard you will have to work on the next topics.
* Updates: Keep your model and libraries updated with the latest security patches.
* Input Sanitation: Before feeding data to the model, sanitize inputs rigorously. This involves techniques such as:
    * Validation: Enforce strict rules on allowed characters and data types.
    * Filtering: Remove potentially malicious scripts or code fragments.
    * Encoding: Convert special characters into safe representations.
    * Verification: Run tooling that identifies potential script injections (e.g. [models that detect prompt injection attempts](https://python.langchain.com/docs/guides/safety/hugging_face_prompt_injection)). 

### Data privacy
To protect sensitive data from potential leaks or unauthorized access, it is essential to sandbox the model execution. This means running the model in a secure, isolated environment, which helps mitigate many attack vectors.

When training the model with sensitive data, expose your newly-trained model to tests to identify potential sensitive data leaks.

### Untrusted environments or networks

If you can't run your models in a secure and isolated environment or if it must be exposed to an untrusted network, make sure to take the following security precautions:
* Confirm the hash of  any downloaded artifact (i.e. pre-trained model weights) matches a known-good value
* Encrypt your data while sending it over the network.

### Multi-Tenant environments

If you intend to run multiple models in parallel with shared memory, it is your responsibility to ensure the models do not interact or access each other's data. The primary areas of concern are tenant isolation, resource allocation, model sharing and hardware attacks.

#### Tenant Isolation

You must make sure that models run separately. Since models can run code, it's important to use strong isolation methods to prevent unwanted access to the data from other tenants.

Separating networks is also a big part of isolation. If you keep model network traffic separate, you not only prevent unauthorized access to data or models, but also prevent malicious users or tenants sending graphs to execute under another tenant’s identity.

#### Resource Allocation

A denial of service caused by one model can impact the overall system health. Implement safeguards like rate limits, access controls, and health monitoring.

#### Model Sharing

In a multitenant design that allows sharing models, make sure that tenants and users fully understand the potential security risks involved. They must be aware that they will essentially be running code provided by other users. Unfortunately, there are no reliable methods available to detect malicious models, graphs, or checkpoints. To mitigate this risk, the recommended approach is to sandbox the model execution, effectively isolating it from the rest of the system.

#### Hardware Attacks

Besides the virtual environment, the hardware (GPUs or TPUs) can also be attacked. [Research](https://scholar.google.com/scholar?q=gpu+side+channel) has shown that side channel attacks on GPUs are possible, which can make data leak from other models or processes running on the same system at the same time.

## Reporting a Vulnerability

Beware that none of the topics under [Using Keras Securely](#using-keras-securely) are considered vulnerabilities of Keras.

If you have discovered a security vulnerability in this project, please report it
privately. **Do not disclose it as a public issue.** This gives us time to work with you
to fix the issue before public exposure, reducing the chance that the exploit will be
used before a patch is released.

You may submit the report in the following ways:

- send an email to francois.chollet@gmail.com; and/or
- send a [private vulnerability report](https://github.com/keras-team/keras/security/advisories/new)

Please provide the following information in your report:

- A description of the vulnerability and its impact
- How to reproduce the issue

This project is maintained by volunteers on a reasonable-effort basis. As such,
please give us 90 days to work on a fix before public exposure.
