FROM rcorbish/openblas-jre9

WORKDIR /home/remix

ADD run.sh  run.sh
ADD src/main/resources  /home/remix/resources
ADD target/classes  /home/remix/classes
ADD target/dependency /home/remix/libs

RUN chmod 0500 run.sh ; \
	sed -i "s/\${VERSION}/$(date)/g" resources/templates/layout.jade
ENV CP classes:resources

VOLUME [ "/home/remix/data" ]

ENTRYPOINT [ "sh", "/home/remix/run.sh" ]  
CMD [ "data" ]
