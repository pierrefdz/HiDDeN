import numpy as np
import torch
import torch.nn as nn

from utils_train import VGGLoss
from noise_layers.noiser import Noiser


class HiDDenConfiguration():
    """
    The HiDDeN network configuration.
    """

    def __init__(self, message_length: int,
                 encoder_blocks: int, encoder_channels: int,
                 decoder_blocks: int, decoder_channels: int,
                 use_discriminator: bool,
                 use_vgg: bool,
                 discriminator_blocks: int, discriminator_channels: int,
                 decoder_loss: float,
                 encoder_loss: float,
                 adversarial_loss: float,
                 enable_fp16: bool = False,
                 lr_enc_dec: float = 0.001,
                 lr_discrim: float = 0.001
                 ):
        self.message_length = message_length
        self.encoder_blocks = encoder_blocks
        self.encoder_channels = encoder_channels
        self.use_discriminator = use_discriminator
        self.use_vgg = use_vgg
        self.decoder_blocks = decoder_blocks
        self.decoder_channels = decoder_channels
        self.discriminator_blocks = discriminator_blocks
        self.discriminator_channels = discriminator_channels
        self.decoder_loss = decoder_loss
        self.encoder_loss = encoder_loss
        self.adversarial_loss = adversarial_loss
        self.enable_fp16 = enable_fp16
        self.lr_enc_dec = lr_enc_dec
        self.lr_discrim = lr_discrim



class ConvBNRelu(nn.Module):
    """
    Building block used in HiDDeN network. Is a sequence of Convolution, Batch Normalization, and ReLU activation
    """
    def __init__(self, channels_in, channels_out):

        super(ConvBNRelu, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, 3, stride=1, padding=1),
            nn.BatchNorm2d(channels_out, eps=1e-3),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.layers(x)


class Encoder(nn.Module):
    """
    Inserts a watermark into an image.
    """
    def __init__(self, config: HiDDenConfiguration):
        super(Encoder, self).__init__()
        self.num_blocks = config.encoder_blocks

        layers = [ConvBNRelu(3, config.encoder_channels)]

        for _ in range(config.encoder_blocks-1):
            layer = ConvBNRelu(config.encoder_channels, config.encoder_channels)
            layers.append(layer)

        self.conv_bns = nn.Sequential(*layers)
        self.after_concat_layer = ConvBNRelu(config.encoder_channels + 3 + config.message_length, config.encoder_channels)

        self.final_layer = nn.Conv2d(config.encoder_channels, 3, kernel_size=1)

    def forward(self, image, message):

        expanded_message = message.unsqueeze(-1).unsqueeze(-1) # BxLx1x1
        expanded_message = expanded_message.expand(-1,-1, image.size(-2), image.size(-1)) # BxLxHxW

        encoded_image = self.conv_bns(image)
        concat = torch.cat([expanded_message, encoded_image, image], dim=1)
        im_w = self.after_concat_layer(concat)
        im_w = self.final_layer(im_w)
        return im_w


class Decoder(nn.Module):
    """
    Decoder module. Receives a watermarked image and extracts the watermark.
    The input image may have various kinds of noise applied to it,
    such as Crop, JpegCompression, and so on. See Noise layers for more.
    """
    def __init__(self, config: HiDDenConfiguration):

        super(Decoder, self).__init__()
        self.channels = config.decoder_channels

        layers = [ConvBNRelu(3, self.channels)]
        for _ in range(config.decoder_blocks - 1):
            layers.append(ConvBNRelu(self.channels, self.channels))

        layers.append(ConvBNRelu(self.channels, config.message_length))
        layers.append(nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        self.layers = nn.Sequential(*layers)

        self.linear = nn.Linear(config.message_length, config.message_length)

    def forward(self, image_with_wm):
        x = self.layers(image_with_wm) # BxDcx1x1
        x = x.squeeze(-1).squeeze(-1) # BxDc
        x = self.linear(x)
        return x


class EncoderDecoder(nn.Module):
    """
    Combines Encoder->Noiser->Decoder into single pipeline.
    The input is the cover image and the watermark message. The module inserts the watermark into the image
    (obtaining encoded_image), then applies Noise layers (obtaining noised_image), then passes the noised_image
    to the Decoder which tries to recover the watermark (called decoded_message). The module outputs
    a three-tuple: (encoded_image, noised_image, decoded_message)
    """
    def __init__(self, config: HiDDenConfiguration, noiser: Noiser):

        super(EncoderDecoder, self).__init__()
        self.encoder = Encoder(config)
        self.noiser = noiser

        self.decoder = Decoder(config)

    def forward(self, image, message, noise=True):
        encoded_image = self.encoder(image, message)
        if noise:
            noised_and_cover = self.noiser([encoded_image, image])
            noised_image = noised_and_cover[0]
            decoded_message = self.decoder(noised_image)
            return encoded_image, noised_image, decoded_message
        else:
            decoded_message = self.decoder(encoded_image)
            return encoded_image, decoded_message


class Discriminator(nn.Module):
    """
    Discriminator network. Receives an image and has to figure out whether it has a watermark inserted into it, or not.
    """
    def __init__(self, config: HiDDenConfiguration):
        super(Discriminator, self).__init__()

        layers = [ConvBNRelu(3, config.discriminator_channels)]
        for _ in range(config.discriminator_blocks-1):
            layers.append(ConvBNRelu(config.discriminator_channels, config.discriminator_channels))

        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.conv_bns = nn.Sequential(*layers)
        self.linear = nn.Linear(config.discriminator_channels, 2)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, image):
        X = self.conv_bns(image)
        X = self.avg_pool(X) # BxDcx1x1
        X = X.squeeze(-1).squeeze(-1) # BxDc
        X = self.linear(X) # Bx2
        return self.softmax(X)[:,0:1] # Bx1 


class Hidden:
    def __init__(self, configuration: HiDDenConfiguration, device: torch.device, noiser: Noiser):
        """
        :param configuration: Configuration for the net, such as the size of the input image, number of channels in the intermediate layers, etc.
        :param device: torch.device object, CPU or GPU
        :param noiser: Object representing stacked noise layers.
        """
        super(Hidden, self).__init__()

        self.encoder_decoder = EncoderDecoder(configuration, noiser).to(device)
        self.discriminator = Discriminator(configuration).to(device)
        self.optimizer_enc_dec = torch.optim.Adam(self.encoder_decoder.parameters(), lr = configuration.lr_enc_dec)
        self.optimizer_discrim = torch.optim.Adam(self.discriminator.parameters(), lr = configuration.lr_discrim)

        if configuration.use_vgg:
            self.vgg_loss = VGGLoss(3, 1, False)
            self.vgg_loss.to(device)
        else:
            self.vgg_loss = None

        self.config = configuration
        self.device = device

        self.bce_loss = nn.BCELoss().to(device)
        self.mse_loss = nn.MSELoss().to(device)

        # Defined the labels used for training the discriminator/adversarial loss
        self.cover_label = 1
        self.encoded_label = 0


    def train_on_batch(self, batch: list):
        """
        Trains the network on a single batch consisting of images and messages
        :param batch: batch of training data, in the form [images, messages]
        :return: dictionary of error metrics from Encoder, Decoder, and Discriminator on the current batch
        """
        images, messages = batch

        batch_size = images.shape[0]
        self.encoder_decoder.train()
        self.discriminator.train()
        with torch.enable_grad():
            # ---------------- Train the discriminator -----------------------------
            self.optimizer_discrim.zero_grad()
            d_target_label_cover = torch.full((batch_size, 1), self.cover_label, device=self.device, requires_grad=False, dtype=torch.float32)
            d_target_label_encoded = torch.full((batch_size, 1), self.encoded_label, device=self.device, requires_grad=False, dtype=torch.float32)
            g_target_label_encoded = torch.full((batch_size, 1), self.cover_label, device=self.device, requires_grad=False, dtype=torch.float32)

            # train on cover
            d_on_cover = self.discriminator(images)
            d_loss_on_cover = self.bce_loss(d_on_cover, d_target_label_cover)
            d_loss_on_cover.backward()

            # train on fake
            encoded_images, noised_images, decoded_messages = self.encoder_decoder(images, messages)
            d_on_encoded = self.discriminator(encoded_images.detach())
            d_loss_on_encoded = self.bce_loss(d_on_encoded, d_target_label_encoded)
            d_loss_on_encoded.backward()
            
            self.optimizer_discrim.step()

            # --------------Train the generator (encoder-decoder) ---------------------
            self.optimizer_enc_dec.zero_grad()
            d_on_encoded_for_enc = self.discriminator(encoded_images)
            g_loss_adv = self.bce_loss(d_on_encoded_for_enc, g_target_label_encoded)

            if self.vgg_loss == None:
                g_loss_enc = self.mse_loss(encoded_images, images)
            else:
                vgg_on_cov = self.vgg_loss(images)
                vgg_on_enc = self.vgg_loss(encoded_images)
                g_loss_enc = self.mse_loss(vgg_on_cov, vgg_on_enc)

            g_loss_dec = self.mse_loss(decoded_messages, messages)
            g_loss = self.config.adversarial_loss * g_loss_adv + self.config.encoder_loss * g_loss_enc + self.config.decoder_loss * g_loss_dec

            g_loss.backward()
            self.optimizer_enc_dec.step()

        decoded_rounded = decoded_messages.detach().cpu().numpy().round().clip(0, 1)
        bitwise_avg_err = np.sum(np.abs(decoded_rounded - messages.detach().cpu().numpy())) / (batch_size * messages.shape[1])

        losses = {
            'loss': g_loss.item(),
            'encoder_mse': g_loss_enc.item(),
            'dec_mse': g_loss_dec.item(),
            'bitwise-error': bitwise_avg_err,
            'adversarial_bce': g_loss_adv.item(),
            'discr_cover_bce': d_loss_on_cover.item(),
            'discr_encod_bce': d_loss_on_encoded.item()
        }
        return losses, (encoded_images, noised_images, decoded_messages)

    def validate_on_batch(self, batch: list):
        """
        Runs validation on a single batch of data consisting of images and messages
        :param batch: batch of validation data, in form [images, messages]
        :return: dictionary of error metrics from Encoder, Decoder, and Discriminator on the current batch
        """
        images, messages = batch

        batch_size = images.shape[0]

        self.encoder_decoder.eval()
        self.discriminator.eval()
        with torch.no_grad():
            d_target_label_cover = torch.full((batch_size, 1), self.cover_label, device=self.device, requires_grad=False, dtype=torch.float32)
            d_target_label_encoded = torch.full((batch_size, 1), self.encoded_label, device=self.device, requires_grad=False, dtype=torch.float32)
            g_target_label_encoded = torch.full((batch_size, 1), self.cover_label, device=self.device, requires_grad=False, dtype=torch.float32)

            d_on_cover = self.discriminator(images)
            d_loss_on_cover = self.bce_loss(d_on_cover, d_target_label_cover)

            encoded_images, decoded_messages = self.encoder_decoder(images, messages, noise=False)

            d_on_encoded = self.discriminator(encoded_images)
            d_loss_on_encoded = self.bce_loss(d_on_encoded, d_target_label_encoded)

            d_on_encoded_for_enc = self.discriminator(encoded_images)
            g_loss_adv = self.bce_loss(d_on_encoded_for_enc, g_target_label_encoded)

            if self.vgg_loss is None:
                g_loss_enc = self.mse_loss(encoded_images, images)
            else:
                vgg_on_cov = self.vgg_loss(images)
                vgg_on_enc = self.vgg_loss(encoded_images)
                g_loss_enc = self.mse_loss(vgg_on_cov, vgg_on_enc)

            g_loss_dec = self.mse_loss(decoded_messages, messages)
            g_loss = self.config.adversarial_loss * g_loss_adv + self.config.encoder_loss * g_loss_enc + self.config.decoder_loss * g_loss_dec

        decoded_rounded = decoded_messages.detach().cpu().numpy().round().clip(0, 1)
        bitwise_avg_err = np.sum(np.abs(decoded_rounded - messages.detach().cpu().numpy())) / (batch_size * messages.shape[1])

        losses = {
            'loss': g_loss.item(),
            'encoder_mse': g_loss_enc.item(),
            'dec_mse': g_loss_dec.item(),
            'bitwise-error': bitwise_avg_err,
            'adversarial_bce': g_loss_adv.item(),
            'discr_cover_bce': d_loss_on_cover.item(),
            'discr_encod_bce': d_loss_on_encoded.item()
        }
        return losses, (encoded_images, decoded_messages)

    def to_string(self):
        return '{}\n{}'.format(str(self.encoder_decoder), str(self.discriminator))
