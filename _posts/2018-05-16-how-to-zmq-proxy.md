---
title: ZeroMQ Proxy
description: >-
  How to Solve the Dynamic Discovery Problem in ZeroMQ
date: 2018-05-16
categories: [Blog, Tutorial]
tags: [Java, ZeroMQ, Microservices]
pin: true
author: ks
---

[ZeroMQ](https://zeromq.org/) is my favorite message passing and networking library. It has bindings for almost all major languages and it's super convenient to build polyglot distributed network applications with it. Also, ZeroMQ documentation and examples are very exhaustive.

## Dynamic Discovery Problem

I was recently struck with a problem where I had multiple ZeroMQ consumers connecting to multiple ZeroMQ producers. I needed a way to have a static port for consuming messages since adding new consumers and producers was getting hard to maintain. This is because all consumers needed to change whenever a new producer was added to the system.

That's where I discovered the [Dynamic Discovery Problem](https://zguide.zeromq.org/java:chapter2#The-Dynamic-Discovery-Problem), which is precisely the problem that I described above. ZeroMQ documentation suggests using pub-sub proxy as a simple solution to this problem. In the post, I will create a working example for pub-sub proxy in `Java`.

<img src="/assets/zmq_proxy.png" alt="ZMQ Proxy" style="background: white; padding: 10px;">

## Publisher

Let's create a **Publisher** that randomly sends payloads with topics between `0-9`. The publisher will also get a randomly generated server number. Then, we can start multiple of these publishers and connect them to the **Proxy** or **Forwarder** socket.

```java
Context context = ZMQ.context(1);
Socket publisher = context.socket(ZMQ.PUB);
Random rand = new Random(System.currentTimeMillis());
int serverNo = rand.nextInt(10000);
publisher.connect("tcp://127.0.0.1:9999"); // Connect to Proxy Server
```

> The `SUBSCRIBER` endpoint of the Proxy is running on port `9999`

Here is the complete code for **Publisher**

```java
package zmq_forwarder;

import java.util.Random;

import org.zeromq.ZMQ;
import org.zeromq.ZMQ.Context;
import org.zeromq.ZMQ.Socket;

public class Publisher {
	public static void main(String[] args) throws Exception {
		Context context = ZMQ.context(1);
		Socket publisher = context.socket(ZMQ.PUB);

		Random rand = new Random(System.currentTimeMillis());
		int serverNo = rand.nextInt(10000);

		publisher.connect("tcp://127.0.0.1:9999");

		System.out.println(String.format("Server : %s", serverNo));

		try {
			while (!Thread.currentThread().isInterrupted()) {
				String topic = String.format("%s", rand.nextInt(10));
				String payload = String.format("Server#%s", serverNo);
				publisher.sendMore(topic);
				publisher.send(payload);
				System.out.println("Sending: " + payload + " on Channel " + topic);
				Thread.sleep(250);
			}
		} catch (Exception e) {
			System.err.println(e.getMessage());
		} finally {
			publisher.close();
			context.term();
		}
	}
}
```

## Subscriber

Now, we can create a **Subcriber** socket that subscribes to a specific topic (`1` in this case). We connect the subscriber to the Proxy server on a static port.

> The `PUBLISHER` endpoint of the Proxy is running on port `8888`

```java
Context context = ZMQ.context(1);
Socket subscriber = context.socket(ZMQ.SUB);
subscriber.connect("tcp://127.0.0.1:8888");
subscriber.subscribe("1");
```

Now we have all producers and consumers connecting to their respective static ports and **Proxy** will automatically route the producers to consumers. ZeroMQ also supports **Subscription Forwarding**. Hence, whatever topic consumer subscribes to automatically gets forwarded to producer sockets.

> This subscriber socket will receive all messages with topic `1` from all publishers.

Here is the complete code for **Subscriber**

```java
package zmq_forwarder;

import org.zeromq.ZMQ;
import org.zeromq.ZMQ.Context;
import org.zeromq.ZMQ.Socket;

public class Subscriber {
	public static void main(String[] args) {
		Context context = ZMQ.context(1);
		Socket subscriber = context.socket(ZMQ.SUB);

		subscriber.connect("tcp://127.0.0.1:8888");
		subscriber.subscribe("1");

		try {
			while (!Thread.currentThread().isInterrupted()) {
				String topic = subscriber.recvStr();
				String payload = subscriber.recvStr();
				System.out.println(topic + " : " + payload);
			}
		} catch (Exception e) {
			System.err.println(e.getMessage());
		} finally {
			subscriber.close();
			context.term();
		}

	}
}
```

## Proxy

Finally, here we have the Pub-Sub Proxy. As we can see, proxy binds to all producers on port `9999` and all consumers on port `8888`.

```java
Socket frontend = context.socket(ZMQ.SUB);
frontend.bind("tcp://*:9999");
frontend.subscribe("".getBytes());

Socket backend = context.socket(ZMQ.PUB);
backend.bind("tcp://*:8888");
```

Finally, we can create the pub-sub proxy as below.

```java
ZMQ.proxy(frontend, backend, null); // Create Proxy or Forwarder
```

> The `null` third parameter can be another socket where you can sniff the traffic; however, that is a topic for another day.

Here is the complete code for the **Proxy**.

```java
package zmq_forwarder;

import org.zeromq.ZMQ;
import org.zeromq.ZMQ.Context;
import org.zeromq.ZMQ.Socket;

public class Forwarder {
	public static void main(String[] args) {

		Context context = ZMQ.context(1);
		Socket frontend = context.socket(ZMQ.SUB);
		frontend.bind("tcp://*:9999");
		frontend.subscribe("".getBytes());

		Socket backend = context.socket(ZMQ.PUB);
		backend.bind("tcp://*:8888");

		try {
			System.out.println("Starting forwarder");
			ZMQ.proxy(frontend, backend, null);
		} catch (Exception e) {
			System.err.println(e.getMessage());
		} finally {
			frontend.close();
			backend.close();
			context.term();
		}

	}

}
```

<hr>

## Working Project

You can find a `maven` based project for this post on my [Github](https://github.com/kapilsh/mini-projects/tree/master/java/zmq_forwarder)

Build the project by running `mvn clean package` in the terminal. Post build, you should see three `jar` files in the `target` folder:
- `publisher.jar`
- `subscriber.jar`
- `proxy.jar`

> The [POM](https://github.com/kapilsh/mini-projects/blob/master/java/zmq_forwarder/pom.xml) file for the project also shows how to create an executable jar in a maven based project using `maven-assembly-plugin`

```bash
$ mvn clean package
[INFO] Scanning for projects...
[INFO]
[INFO] ---------------------< com.ksharma:zmq_forwarder >----------------------
[INFO] Building zmq_forwarder 0.0.1
[INFO] --------------------------------[ jar ]---------------------------------
[INFO]
[INFO] --- maven-clean-plugin:2.5:clean (default-clean) @ zmq_forwarder ---
[INFO] Deleting /Users/kapilsharma/dev/git/mini-projects/java/zmq_forwarder/target
[INFO]
[INFO] --- maven-resources-plugin:2.6:resources (default-resources) @ zmq_forwarder ---
[INFO] Using 'UTF-8' encoding to copy filtered resources.
[INFO] skip non existing resourceDirectory /Users/kapilsharma/dev/git/mini-projects/java/zmq_forwarder/src/main/resources
[INFO]
[INFO] --- maven-compiler-plugin:3.3:compile (default-compile) @ zmq_forwarder ---
[INFO] Changes detected - recompiling the module!
[INFO] Compiling 3 source files to /Users/kapilsharma/dev/git/mini-projects/java/zmq_forwarder/target/classes
[INFO]
[INFO] --- maven-resources-plugin:2.6:testResources (default-testResources) @ zmq_forwarder ---
[INFO] Using 'UTF-8' encoding to copy filtered resources.
[INFO] skip non existing resourceDirectory /Users/kapilsharma/dev/git/mini-projects/java/zmq_forwarder/src/test/resources
[INFO]
[INFO] --- maven-compiler-plugin:3.3:testCompile (default-testCompile) @ zmq_forwarder ---
[INFO] Nothing to compile - all classes are up to date
[INFO]
[INFO] --- maven-surefire-plugin:2.12.4:test (default-test) @ zmq_forwarder ---
[INFO] No tests to run.
[INFO]
[INFO] --- maven-jar-plugin:2.4:jar (default-jar) @ zmq_forwarder ---
[INFO] Building jar: /Users/kapilsharma/dev/git/mini-projects/java/zmq_forwarder/target/zmq_forwarder-0.0.1.jar
[INFO]
[INFO] --- maven-assembly-plugin:2.5.5:single (publisher) @ zmq_forwarder ---
[INFO] Building jar: /Users/kapilsharma/dev/git/mini-projects/java/zmq_forwarder/target/publisher.jar
[INFO]
[INFO] --- maven-assembly-plugin:2.5.5:single (subscriber) @ zmq_forwarder ---
[INFO] Building jar: /Users/kapilsharma/dev/git/mini-projects/java/zmq_forwarder/target/subscriber.jar
[INFO]
[INFO] --- maven-assembly-plugin:2.5.5:single (proxy) @ zmq_forwarder ---
[INFO] Building jar: /Users/kapilsharma/dev/git/mini-projects/java/zmq_forwarder/target/proxy.jar
[INFO] ------------------------------------------------------------------------
[INFO] BUILD SUCCESS
[INFO] ------------------------------------------------------------------------
[INFO] Total time: 2.892 s
[INFO] Finished at: 2018-05-16T18:22:56-05:00
[INFO] ------------------------------------------------------------------------
```


```bash
$ ls target | grep jar
proxy.jar
publisher.jar
subscriber.jar
zmq_forwarder-0.0.1.jar
```

## My Awesome Tmux Window

Run all the `jar` files as below:

![My Awesome Terminal Window](/assets/forwarder.png)

> Consumer is receiving all messages from both producers for topic `1`

# Conclusion

ZMQ Proxy provides a neat way to connect multiple ZeroMQ subscribers to multiple ZeroMQ publishers and solves the **Dynamic Discovery Problem**
