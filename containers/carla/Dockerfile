FROM carlasim/carla:0.9.10.1
WORKDIR /home/carla
USER root
COPY AdditionalMaps_0.9.10.1.tar.gz .
RUN tar -xf AdditionalMaps_0.9.10.1.tar.gz
RUN rm AdditionalMaps_0.9.10.1.tar.gz
USER carla
CMD ["/bin/bash"]